#!/bin/bash
# Setup Colab Git Authentication with Google Drive persistence
# Author: tungmv
# 
# Usage (Method 1 - Environment Variable - Recommended for Colab):
#   In Python cell:
#     import os
#     os.environ['GITHUB_PAT'] = 'ghp_your_token_here'
#   Then run:
#     !bash scripts/setup_colab_git.sh
#
# Usage (Method 2 - Interactive):
#   !bash scripts/setup_colab_git.sh
#   (Then paste token when prompted)

set -e  # Exit on error

# Constants
DRIVE_CRED_FILE="/content/drive/MyDrive/.colab_git_credentials"
PAT_CREATE_URL="https://github.com/settings/tokens/new?scopes=repo&description=Colab-FL-Sync"
GITHUB_USERNAME="tungmv"
GITHUB_EMAIL="tungminh4399@gmail.com"
REPO_URL="https://github.com/tungmv/fl_clean.git"
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

check_drive_mounted() {
    if [ ! -d "/content/drive/MyDrive" ]; then
        print_error "Google Drive is not mounted"
        echo "   Please mount Drive first:"
        echo "   from google.colab import drive"
        echo "   drive.mount('/content/drive')"
        return 1
    fi
    print_success "Drive mounted at /content/drive"
    return 0
}

create_pat_guidance() {
    echo ""
    print_info "Create GitHub Personal Access Token:"
    echo "  URL: ${PAT_CREATE_URL}"
    echo ""
    echo "  Required settings:"
    echo "    - Scope: repo (Full control of private repositories)"
    echo "    - Description: Colab-FL-Sync"
    echo ""
    echo "  Copy the generated token (starts with 'ghp_' or 'github_pat_')"
    echo ""
}

read_pat_securely() {
    local pat=""
    
    # Check if PAT is provided via environment variable (REQUIRED for Colab)
    if [ -n "$GITHUB_PAT" ]; then
        print_info "Using PAT from environment variable"
        # Trim whitespace
        pat=$(echo "$GITHUB_PAT" | xargs)
        echo "$pat"
        return 0
    fi
    
    # If no environment variable, show clear error message
    print_error "GITHUB_PAT environment variable not set"
    echo ""
    echo "=========================================="
    echo "  SETUP FAILED - MISSING TOKEN"
    echo "=========================================="
    echo ""
    echo "In Google Colab, you MUST set the token as an environment variable."
    echo ""
    echo "Run this Python code first:"
    echo ""
    echo "  import os"
    echo "  os.environ['GITHUB_PAT'] = 'ghp_your_token_here'"
    echo "  !bash scripts/setup_colab_git.sh"
    echo ""
    echo "Replace 'ghp_your_token_here' with your actual GitHub token."
    echo ""
    return 1
}

validate_pat() {
    local pat="$1"
    
    # Debug: show first few characters (for troubleshooting)
    local pat_preview="${pat:0:10}"
    print_info "Validating token: ${pat_preview}..."
    
    # Check if token is empty
    if [ -z "$pat" ]; then
        print_error "Token is empty"
        return 1
    fi
    
    # Check format - token should start with ghp_ or github_pat_
    if [[ "$pat" =~ ^ghp_ ]] || [[ "$pat" =~ ^github_pat_ ]]; then
        print_success "Token format valid"
    else
        print_error "Invalid PAT format. Must start with 'ghp_' or 'github_pat_'"
        echo "  Received: ${pat:0:20}..." # Show first 20 chars for debugging
        return 1
    fi
    
    # Test connection
    print_info "Testing connection to GitHub..."
    if git ls-remote "https://${GITHUB_USERNAME}:${pat}@github.com/tungmv/fl_clean.git" &>/dev/null; then
        print_success "Connection successful"
        local masked_pat="${pat:0:8}...${pat: -4}"
        print_success "PAT validated (${masked_pat})"
        return 0
    else
        print_error "Connection failed. PAT may be invalid or network issue"
        return 1
    fi
}

save_credentials() {
    local pat="$1"
    
    # Create credentials file
    cat > "$DRIVE_CRED_FILE" << EOF
GITHUB_USERNAME=${GITHUB_USERNAME}
GITHUB_EMAIL=${GITHUB_EMAIL}
GITHUB_PAT=${pat}
REPO_URL=${REPO_URL}
BRANCH=${BRANCH}
EOF
    
    # Set secure permissions
    chmod 600 "$DRIVE_CRED_FILE"
    
    print_success "Credentials saved to Drive"
}

configure_git() {
    local pat="$1"
    
    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        print_info "Not in a git repository, skipping git configuration"
        return 0
    fi
    
    # Configure git remote (convert SSH to HTTPS with PAT)
    local https_url="https://${GITHUB_USERNAME}:${pat}@github.com/tungmv/fl_clean.git"
    git remote set-url origin "$https_url" 2>/dev/null || true
    print_success "Git remote configured (HTTPS)"
    
    # Configure git user
    git config user.name "$GITHUB_USERNAME"
    git config user.email "$GITHUB_EMAIL"
    print_success "Git user configured (${GITHUB_USERNAME} <${GITHUB_EMAIL}>)"
}

# Main
main() {
    echo "🔐 Colab Git Setup"
    echo ""
    
    # Check if Drive is mounted
    if ! check_drive_mounted; then
        exit 1
    fi
    
    # Check if credentials already exist
    if [ -f "$DRIVE_CRED_FILE" ]; then
        echo ""
        print_info "Credentials already exist in Drive"
        read -p "   Overwrite? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Setup cancelled."
            exit 0
        fi
    fi
    
    # Guide user to create PAT
    create_pat_guidance
    
    # Read PAT from user
    PAT=$(read_pat_securely)
    
    # Validate PAT
    if ! validate_pat "$PAT"; then
        exit 1
    fi
    
    # Save credentials to Drive
    save_credentials "$PAT"
    
    # Configure git
    configure_git "$PAT"
    
    echo ""
    print_success "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: bash scripts/sync.sh"
    echo "  2. Or use: bash scripts/colab_init.sh (for new sessions)"
    echo ""
}

# Run main
main
