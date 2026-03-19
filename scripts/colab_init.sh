#!/bin/bash
# Initialize Colab environment - mount Drive, clone/sync repo
# Author: tungmv
# Usage: bash scripts/colab_init.sh

set -e  # Exit on error

# Constants
REPO_URL="https://github.com/tungmv/fl_clean.git"
REPO_DIR="/content/fl_clean"
BRANCH="modular"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}❌ Error:${NC} $1" >&2
}

print_info() {
    echo -e "${BLUE}📊${NC} $1"
}

is_colab() {
    [ -d "/content" ]
}

check_drive_mounted() {
    if ! is_colab; then
        # Not in Colab, skip Drive check
        return 0
    fi
    
    if [ ! -d "/content/drive/MyDrive" ]; then
        print_error "Google Drive is not mounted"
        echo ""
        echo "   Please mount Drive first with this Python code:"
        echo ""
        echo "   from google.colab import drive"
        echo "   drive.mount('/content/drive')"
        echo ""
        return 1
    fi
    
    print_success "Drive mounted at /content/drive"
    return 0
}

setup_repository() {
    if [ -d "$REPO_DIR" ]; then
        print_success "Repository found: $REPO_DIR"
        cd "$REPO_DIR"
    else
        echo ""
        print_info "Cloning repository..."
        cd /content
        if ! git clone "$REPO_URL" fl_clean 2>/dev/null; then
            print_error "Failed to clone repository"
            echo "   Check your network connection"
            return 1
        fi
        cd "$REPO_DIR"
        print_success "Repository cloned: $REPO_DIR"
    fi
    
    return 0
}

sync_latest() {
    echo ""
    if [ -f "scripts/sync.sh" ]; then
        bash scripts/sync.sh
        return $?
    else
        print_error "sync.sh not found"
        echo "   Repository may be corrupted"
        return 1
    fi
}

show_environment() {
    echo ""
    print_info "Environment:"
    
    # Python version
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo "   Python: $python_version"
    fi
    
    # Current directory
    echo "   Directory: $(pwd)"
    
    # Current branch
    if [ -d ".git" ]; then
        local current_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
        echo "   Branch: $current_branch"
    fi
    
    # GPU status (if in Colab)
    if is_colab && command -v nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$gpu_name" ]; then
            echo "   GPU: $gpu_name"
        fi
    fi
}

# Main
main() {
    echo "🚀 Initializing Colab environment..."
    echo ""
    
    # Check if Drive is mounted
    if ! check_drive_mounted; then
        exit 1
    fi
    
    # Setup repository (clone if needed, cd to it)
    if ! setup_repository; then
        exit 1
    fi
    
    # Sync latest from GitHub
    if ! sync_latest; then
        exit 1
    fi
    
    # Show environment info
    show_environment
    
    echo ""
    print_success "Ready to work!"
    echo ""
}

# Run main
main
