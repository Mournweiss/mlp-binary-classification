#!/usr/bin/env bash

set -euo pipefail

# ANSI color codes
COLOR_INFO="\033[0m"       # White (default)
COLOR_WARN="\033[1;33m"    # Yellow
COLOR_ERROR="\033[1;31m"   # Red
COLOR_SUCCESS="\033[1;32m" # Green
COLOR_RESET="\033[0m"

info()    { echo -e "${COLOR_INFO}[INFO]${COLOR_RESET} $1" >&2; }
warn()    { echo -e "${COLOR_WARN}[WARN]${COLOR_RESET} $1" >&2; }
error()   { echo -e "${COLOR_ERROR}[ERROR]${COLOR_RESET} $1" >&2; exit 1; }
success() { echo -e "${COLOR_SUCCESS}[OK]${COLOR_RESET} $1" >&2; }

env_file=".env"

# Runtime variables
DOWNLOAD_DATASET=false
FORCE_DOWNLOAD=false
EXEC_MODE="train"

# Parses command-line arguments and sets global variables for orchestrator and options.
#
# Parameters:
# - $@: array - command-line arguments
#
# Returns:
# - None (sets global shell vars)
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --download|-d)
                DOWNLOAD_DATASET=true
                shift
                ;;
            --force|-f)
                FORCE_DOWNLOAD=true
                shift
                ;;
            --mode|-m)
                if [[ -n "$2" && "$2" != "--"* ]]; then
                    EXEC_MODE="$2"
                    shift 2
                else
                    error "Missing mode argument for --mode flag"
                fi
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                warn "Unknown argument: $1"
                shift
                ;;
        esac
    done
}

# Shows help message
show_help() {
    cat <<'EOF'
Usage: $0 [OPTIONS]

Options:
  --download, -d    Download dataset if missing
  --force, -f       Force download even if dataset exists
  --mode, -m <MODE> Execution mode: train, evaluate, predict
  --help, -h        Show this help message

Examples:
  $0 --download
  $0 --mode train --download
  $0
EOF
}

# Ensures all variables from the template are present in the actual env file.
#
# Parameters:
# - template_file: string - path to the template .env file (e.g., .env.example)
# - env_file: string - path to the target .env file
#
# Returns:
# - None
ensure_env_vars() {
    local template_file="$1"
    local env_file="$2"
    local updated=0
    info "Syncing .env with template: $template_file"
    
    # Read all variables from template
    local template_vars=$(read_env_file -f "$template_file")
    
    # Process each variable from the template
    for var in $template_vars; do
        [[ -z "$var" ]] && continue
        
        local var_name="${var%%=*}"
        [[ "$var_name" =~ ^[[:space:]]*# ]] && continue
        
        info "Checking if $var_name is present in $env_file ..."
        if ! grep -Eq "^[[:space:]]*#?[[:space:]]*$var_name[[:space:]]*=" "$env_file"; then
            last_char=$(tail -c1 "$env_file" 2>/dev/null || echo '')
            if [[ "$last_char" != "" && "$last_char" != $'\n' ]]; then
                echo >> "$env_file"
            fi
            echo "$var" >> "$env_file"
            info "Added $var_name to $env_file"
            updated=1
        fi
    done
    
    if [[ $updated -eq 1 ]]; then
        info "Completed variable sync: $env_file updated"
    else
        info "No missing variables detected in $env_file"
    fi
}

# Read environment variables from file and return them as a space-separated string
#
# Parameters:
# - $@: array - command-line arguments for specifying file and variables
#   --file|-f <file>: specify the environment file to read (default: .env)
#   --value|-v: return only values, not key=value pairs
#   other args: specific variable names to read (comma-separated)
#
# Returns:
# - string: space-separated environment variables in key=value format or just values if --value flag is used
read_env_file() {
    local env_file=""
    local vars_to_read=()
    local value_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --value|-v)
                value_only=true
                shift
                ;;
            --file|-f)
                if [[ -n "$2" && "$2" != "--"* ]]; then
                    env_file="$2"
                    shift 2
                else
                    error "Missing file argument for --file flag"
                fi
                ;;
            --file=*)
                env_file="${1#*=}"
                shift
                ;;
            *)
                if [[ -n "$env_file" ]]; then
                    IFS=',' read -ra vars <<< "$1"
                    for var in "${vars[@]}"; do
                        vars_to_read+=("$var")
                    done
                else
                    env_file="$1"
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "$env_file" ]]; then
        env_file=".env"
    fi
    
    if [[ ! -f "$env_file" ]]; then
        info "Environment file $env_file not found"
        return 1
    fi
    
    local env_args=""
    
    # Read and process each line
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Extract variable name and value
        local var_name="${line%%=*}"
        local var_value="${line#*=}"
        
        # Remove trailing comments and trim whitespace
        var_value="${var_value%%#*}"
        var_name="$(echo "$var_name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        var_value="$(echo "$var_value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        
        # Skip if variable name is empty
        [[ -z "$var_name" ]] && continue
        
        # If specific variables are requested, check if this one matches
        if [[ ${#vars_to_read[@]} -gt 0 ]]; then
            local found=false
            for var in "${vars_to_read[@]}"; do
                if [[ "$var" == "$var_name" ]]; then
                    found=true
                    break
                fi
            done
            [[ "$found" == false ]] && continue
        fi
        
        # Handle output format based on value_only flag
        if [[ "$value_only" == true ]]; then
            echo "$var_value"
        else
            # Return key=value pairs
            if [[ -n "$env_args" ]]; then
                env_args="$env_args $var_name=$var_value"
            else
                env_args="$var_name=$var_value"
            fi
        fi
    done < "$env_file"
    
    # Output all variables
    if [[ "$value_only" == false ]]; then
        echo "$env_args"
    fi
}

# Verifies or creates .env file from example.
#
# Parameters:
# - None (uses shell globals)
#
# Returns:
# - None
make_env() {
    if [[ -f .env ]]; then
        info "Using existing .env"
        ensure_env_vars .env.example .env
    else
        [[ -f .env.example ]] || error "No key/.env.example template found"
        info "Creating .env from .env.example..."
        local temp_env=""
        temp_env=$(read_env_file -f .env.example)
        echo "$temp_env" | tr ' ' '\n' > .env
        success "Created .env from .env.example"
    fi
}

# Read and export environment variables into current context
set_env() {
    local env_args=$(read_env_file -f .env)
    
    # Export all variables to current context
    if [[ -n "$env_args" ]]; then
        export $env_args
    fi
}

# Downloads the Titanic dataset repository.
#
# Parameters:
# - None (uses DATASET_URL from env)
#
# Returns:
# - 0 on success, 1 on failure
download_dataset() {
    info "Downloading Titanic dataset..."

    mkdir -p data

    # Build URLs by appending /train.csv and /test.csv to base URL
    local base_url="${DATASET_URL:-https://raw.githubusercontent.com/ashishpatel26/Titanic-Machine-Learning-from-Disaster/master/input}"
    local train_url="${base_url}/train.csv"
    local test_url="${base_url}/test.csv"

    info "Downloading train.csv from: ${train_url}"
    if command -v curl &>/dev/null; then
        if ! curl -fSL -o "data/train.csv" "$train_url" 2>/dev/null; then
            error "Failed to download train.csv"
        fi
    elif command -v wget &>/dev/null; then
        if ! wget -q -O "data/train.csv" "$train_url" 2>/dev/null; then
            error "Failed to download train.csv"
        fi
    else
        error "Neither curl nor wget found. Cannot download dataset"
    fi

    info "Downloading test.csv from: ${test_url}"
    if command -v curl &>/dev/null; then
        if ! curl -fSL -o "data/test.csv" "$test_url" 2>/dev/null; then
            error "Failed to download test.csv"
        fi
    elif command -v wget &>/dev/null; then
        if ! wget -q -O "data/test.csv" "$test_url" 2>/dev/null; then
            error "Failed to download test.csv"
        fi
    else
        error "Neither curl nor wget found. Cannot download dataset"
    fi

    if [[ -f "data/train.csv" && -f "data/test.csv" ]]; then
        success "Dataset download completed successfully"
        return 0
    fi

    error "Dataset download failed"
}

# Validates that train.csv and test.csv exist in the data directory.
#
# Parameters:
# - None (uses TRAIN_DATA_PATH, TEST_DATA_PATH from env or defaults)
#
# Returns:
# - 0 if both files exist, 1 otherwise
validate_dataset() {
    local train_file="${TRAIN_DATA_PATH:-data/train.csv}"
    local test_file="${TEST_DATA_PATH:-data/test.csv}"

    local train_exists=false
    local test_exists=false

    if [[ -f "$train_file" ]]; then
        train_exists=true
        local train_rows
        train_rows=$(wc -l < "$train_file")
        info "  train.csv: ${train_rows} lines"
    fi

    if [[ -f "$test_file" ]]; then
        test_exists=true
        local test_rows
        test_rows=$(wc -l < "$test_file")
        info "  test.csv: ${test_rows} lines"
    fi

    if $train_exists && $test_exists; then
        success "Dataset validation passed"
        return 0
    elif $train_exists; then
        warn "  test.csv is missing"
        return 1
    elif $test_exists; then
        warn "  train.csv is missing"
        return 1
    else
        warn "  Neither train.csv nor test.csv found"
        return 1
    fi
}

# Runs the Python script with the specified mode and arguments.
#
# Parameters:
# - $1: string - execution mode (train, evaluate, predict)
# - $@: array - additional arguments to pass to the Python script
#
# Returns:
# - None
run_project() {
    local mode="${EXEC_MODE:-train}"

    info "============================================"
    info "  Titanic MLP — Execution Mode: ${mode}"
    info "============================================"

    case "$mode" in
        train)
            pixi run --environment default train
            ;;
        evaluate)
            pixi run --environment default train-eval
            ;;
        predict)
            pixi run --environment default train-predict
            ;;
        *)
            error "Unknown mode: $mode"
            ;;
    esac
}

# Checks and installs/updates project dependencies.
#
# Parameters:
# - None
#
# Returns:
# - None (exits on failure)
check_dependencies() {
    # Install/update dependencies
    info "Installing/updating dependencies..."
    if ! pixi install --environment default; then
        error "Failed to install dependencies"
    fi
    success "Dependencies are up to date"
}

# Main orchestration entrypoint
#
# Parameters:
# - $@: array - command-line invocation arguments
#
# Returns:
# - None
main() {
    parse_args "$@"
    check_dependencies
    make_env
    set_env

    # Validate or download dataset
    if $DOWNLOAD_DATASET || $FORCE_DOWNLOAD; then
        if ! validate_dataset; then
            download_dataset
        else
            success "Dataset already exists. Use --force to re-download"
        fi
    else
        if ! validate_dataset; then
            warn "Dataset not found. Run './build.sh --download' to download"
        fi
    fi

    run_project
}

main "$@"