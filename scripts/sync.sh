#!/bin/bash
# Sync from GitHub - Pull latest commits
# Author: tungmv
# Usage: bash scripts/sync.sh

set -e  # Exit on error

# Constants
DRIVE_CRED_FILE="/content/drive/MyDrive/.colab_git_credentials"
BRANCH="modular"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}❌ Error:${NC} $1" >&2
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

is_colab() {
    [ -d "/content" ]
}

check_drive_mounted() {
    if is_colab && [ ! -d "/content/drive/MyDrive" ]; then
        print_error "Google Drive is not mounted"
        echo "   Please mount Drive first:"
        echo "   from google.colab import drive"
        echo "   drive.mount('/content/drive')"
        return 1
    fi
    return 0
}

load_credentials() {
    if [ ! -f "$DRIVE_CRED_FILE" ]; then
        print_error "Credentials not found in Drive"
        echo "   Running setup..."
        echo ""
        bash scripts/setup_colab_git.sh
        return $?
    fi
    
    # Load credentials
    source "$DRIVE_CRED_FILE"
    
    # Validate required variables
    if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_PAT" ] || [ -z "$REPO_URL" ]; then
        print_error "Invalid credentials file"
        echo "   Please run: bash scripts/setup_colab_git.sh"
        return 1
    fi
    
    return 0
}

configure_git_auth() {
    # Configure git remote with PAT
    local https_url="https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/tungmv/fl_clean.git"
    git remote set-url origin "$https_url" 2>/dev/null || true
    
    # Configure git user
    git config user.name "$GITHUB_USERNAME" 2>/dev/null || true
    git config user.email "$GITHUB_EMAIL" 2>/dev/null || true
}

check_git_repository() {
    if [ ! -d ".git" ]; then
        print_error "Not in a git repository"
        echo "   cd to the repository directory first"
        return 1
    fi
    return 0
}

check_uncommitted_changes() {
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        print_error "You have uncommitted changes"
        echo "   Run: git status"
        echo "   Commit or stash your changes before syncing"
        return 1
    fi
    return 0
}

fetch_and_pull() {
    # Fetch from remote
    if ! git fetch origin "$BRANCH" 2>/dev/null; then
        print_error "Failed to fetch from GitHub"
        echo "   Check your network connection"
        return 1
    fi
    
    # Check commits behind
    local commits_behind=$(git rev-list HEAD..origin/$BRANCH --count 2>/dev/null || echo "0")
    
    if [ "$commits_behind" -eq 0 ]; then
        print_success "Already up to date!"
        return 0
    fi
    
    print_info "Pulling $commits_behind commit(s)..."
    
    # Pull with fast-forward only
    if ! git pull --ff-only origin "$BRANCH" 2>/dev/null; then
        print_error "Pull failed (fast-forward not possible)"
        echo "   You may have diverged commits"
        echo "   Run: git status"
        return 1
    fi
    
    print_success "Pulled successfully"
    print_success "Up to date! ($commits_behind commit(s) pulled)"
    
    return 0
}

# Main
main() {
    echo "🔄 Syncing from GitHub..."
    
    # Check if Drive is mounted (in Colab)
    if is_colab; then
        if ! check_drive_mounted; then
            exit 1
        fi
        print_success "Drive mounted"
    fi
    
    # Check if in git repository
    if ! check_git_repository; then
        exit 1
    fi
    
    # Load credentials (in Colab) or skip (local)
    if is_colab; then
        if ! load_credentials; then
            exit 1
        fi
        print_success "Credentials loaded"
        configure_git_auth
    fi
    
    # Check for uncommitted changes
    if ! check_uncommitted_changes; then
        exit 1
    fi
    
    # Fetch and pull
    if ! fetch_and_pull; then
        exit 1
    fi
    
    echo "✅ Sync complete!"
}

# Run main
main
