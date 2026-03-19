# Colab Auto-Sync Scripts

Automated scripts to sync your GitHub repository with Google Colab using Google Drive for credential persistence.

## Quick Start

### First Time Setup (One-time)

1. **Mount Google Drive in Colab:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Clone the repository:**
   ```bash
   !git clone https://github.com/tungmv/fl_clean.git
   %cd fl_clean
   ```

3. **Run setup to configure authentication:**
   ```bash
   !bash scripts/setup_colab_git.sh
   ```
   
   This will:
   - Guide you to create a GitHub Personal Access Token (PAT)
   - Save your credentials to Google Drive
   - Configure git authentication

### Every New Colab Session

Use the one-command init script:

```python
from google.colab import drive
drive.mount('/content/drive')

!bash scripts/colab_init.sh
```

This automatically:
- Checks if repository exists (clones if needed)
- Loads credentials from Google Drive
- Pulls latest changes from GitHub
- Shows environment info

### Manual Sync During Session

To pull latest changes anytime:

```bash
%cd /content/fl_clean
!bash scripts/sync.sh
```

---

## Scripts Overview

### 1. `setup_colab_git.sh`

**Purpose:** One-time setup to create and store GitHub credentials

**Usage:**
```bash
bash scripts/setup_colab_git.sh
```

**What it does:**
- Opens GitHub PAT creation URL with pre-configured settings
- Prompts you to paste your PAT
- Validates PAT by testing connection
- Saves credentials to `/content/drive/MyDrive/.colab_git_credentials`
- Configures git remote (converts SSH → HTTPS with PAT)
- Configures git user (name and email)

**When to use:**
- First time using the repo in Colab
- When PAT expires or needs to be changed
- When credentials are corrupted

---

### 2. `sync.sh`

**Purpose:** Pull latest commits from GitHub

**Usage:**
```bash
bash scripts/sync.sh
```

**What it does:**
- Loads credentials from Google Drive
- Checks for uncommitted local changes
- Fetches latest from `origin/modular`
- Pulls changes with fast-forward only
- Shows minimal output (success/error)

**Output examples:**

Success:
```
🔄 Syncing from GitHub...
✓ Drive mounted
✓ Credentials loaded
✓ Pulling 3 commit(s)...
✓ Pulled successfully
✓ Up to date! (3 commit(s) pulled)
✅ Sync complete!
```

Already up-to-date:
```
🔄 Syncing from GitHub...
✓ Drive mounted
✓ Credentials loaded
✓ Already up to date!
✅ Sync complete!
```

Error (uncommitted changes):
```
🔄 Syncing from GitHub...
✓ Drive mounted
✓ Credentials loaded
❌ Error: You have uncommitted changes
   Run: git status
   Commit or stash your changes before syncing
```

**When to use:**
- Before starting work (get latest code)
- After someone pushes changes to GitHub
- To check if you're behind remote

---

### 3. `colab_init.sh`

**Purpose:** Complete initialization for new Colab sessions

**Usage:**
```bash
bash scripts/colab_init.sh
```

**What it does:**
- Checks if Google Drive is mounted
- Checks if repository exists at `/content/fl_clean`
  - If not: clones from GitHub
  - If exists: navigates to it
- Runs `sync.sh` to pull latest changes
- Shows environment summary (Python version, branch, GPU)

**Output example:**
```
🚀 Initializing Colab environment...

✓ Drive mounted at /content/drive
✓ Repository found: /content/fl_clean
✓ Branch: modular

🔄 Syncing from GitHub...
✓ Credentials loaded
✓ Pulled 2 commits
✅ Sync complete!

📊 Environment:
   Python: 3.10.12
   Directory: /content/fl_clean
   Branch: modular
   GPU: Tesla T4

✅ Ready to work!
```

**When to use:**
- Every new Colab session (recommended)
- Quick one-command setup

---

## GitHub Personal Access Token (PAT)

### Creating a PAT

1. Go to: https://github.com/settings/tokens/new
2. Set description: `Colab-FL-Sync`
3. Select scope: **`repo`** (Full control of private repositories)
4. Set expiration: Choose your preference (30/60/90 days or no expiration)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again)

### PAT Format

Valid PAT formats:
- Classic: `ghp_xxxxxxxxxxxxxxxxxxxx` (40 characters)
- Fine-grained: `github_pat_xxxxxxxxxxxxxxxxxxxx`

### Revoking a PAT

If your PAT is compromised:

1. Go to: https://github.com/settings/tokens
2. Find "Colab-FL-Sync" token
3. Click "Delete" or "Revoke"
4. Run `bash scripts/setup_colab_git.sh` to create a new one

---

## Credential Storage

### Location

Credentials are stored in Google Drive at:
```
/content/drive/MyDrive/.colab_git_credentials
```

### File Format

```bash
GITHUB_USERNAME=tungmv
GITHUB_EMAIL=tungminh4399@gmail.com
GITHUB_PAT=ghp_xxxxxxxxxxxxxxxxxxxx
REPO_URL=https://github.com/tungmv/fl_clean.git
BRANCH=modular
```

### Security Notes

- File has permissions `600` (owner read/write only)
- File is hidden (starts with `.`)
- Stored in your personal Google Drive
- **Never commit this file to git** (already in `.gitignore`)

### Backup

You can manually backup the credentials file from Google Drive. To restore:
1. Place the file back at `/content/drive/MyDrive/.colab_git_credentials`
2. Run `bash scripts/sync.sh` (will auto-load)

---

## Troubleshooting

### Error: "Drive not mounted"

**Solution:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Error: "Credentials not found"

**Solution:**
```bash
bash scripts/setup_colab_git.sh
```

### Error: "You have uncommitted changes"

**Solution:**
```bash
# Check what changed
git status

# Option 1: Commit changes
git add .
git commit -m "your message"

# Option 2: Stash changes
git stash

# Then sync again
bash scripts/sync.sh
```

### Error: "Pull failed (fast-forward not possible)"

This means your local branch has diverged from remote.

**Solution:**
```bash
# See what's different
git status
git log origin/modular..HEAD

# Option 1: Stash local commits, pull, reapply
git stash
git pull origin modular
git stash pop

# Option 2: Hard reset (DANGER: loses local commits)
git fetch origin
git reset --hard origin/modular
```

### Error: "Connection failed" or "Network error"

**Causes:**
- Invalid PAT
- PAT expired
- Network issues in Colab
- GitHub is down

**Solution:**
1. Check GitHub status: https://www.githubstatus.com
2. Try again after a few seconds
3. If persists, regenerate PAT: `bash scripts/setup_colab_git.sh`

### Scripts don't run / Permission denied

**Solution:**
```bash
chmod +x scripts/*.sh
```

---

## Advanced Usage

### Working with Branches

The scripts default to `modular` branch. To change:

**Option 1: Edit credentials file**
```bash
# Edit the BRANCH variable in:
# /content/drive/MyDrive/.colab_git_credentials
BRANCH=your-branch-name
```

**Option 2: Manual checkout**
```bash
git checkout your-branch-name
bash scripts/sync.sh
```

### Syncing from Upstream

If you want to sync from the upstream repo (evgafanboi/fl_clean):

```bash
git fetch upstream
git merge upstream/modular
```

Or add to your workflow:
```bash
# Pull from your fork
bash scripts/sync.sh

# Then pull from upstream
git fetch upstream
git merge upstream/modular
```

### Working with Multiple Repositories

Create separate credential files:

```bash
/content/drive/MyDrive/.colab_git_credentials_project1
/content/drive/MyDrive/.colab_git_credentials_project2
```

Edit scripts to use different credential file paths.

---

## Best Practices

### Recommended Workflow

1. **Start of session:**
   ```bash
   bash scripts/colab_init.sh
   ```

2. **During work:**
   - Make changes, test code
   - Commit locally if desired
   
3. **Sync periodically:**
   ```bash
   bash scripts/sync.sh
   ```

4. **End of session:**
   - Commit important changes
   - Push to GitHub (if desired)

### Avoid Conflicts

- Always sync before making changes
- Commit frequently
- Don't edit same files on multiple machines simultaneously
- Use branches for experimental work

### Security Best Practices

- Use fine-grained PAT with minimal permissions (if available)
- Set PAT expiration (30-90 days)
- Revoke PAT when no longer needed
- Don't share your credential file
- Don't commit `.colab_git_credentials` to git (already in `.gitignore`)

---

## FAQ

**Q: Do I need to run setup every Colab session?**

A: No! Setup is one-time only. Credentials persist in Google Drive across sessions.

**Q: Can I use SSH keys instead of PAT?**

A: No. Colab sessions are ephemeral and don't persist SSH keys. PAT with HTTPS is recommended.

**Q: What if my PAT expires?**

A: Run `bash scripts/setup_colab_git.sh` again to create and save a new PAT.

**Q: Can I use these scripts outside of Colab?**

A: Yes! The scripts detect if running locally and skip Drive-related operations.

**Q: How do I push changes from Colab to GitHub?**

A: The scripts currently only pull. To push:
```bash
git add .
git commit -m "your message"
git push origin modular
```

**Q: What if I accidentally deleted the credential file?**

A: Just run `bash scripts/setup_colab_git.sh` again to recreate it.

**Q: Can I use this for private repositories?**

A: Yes! Make sure your PAT has `repo` scope (full control).

---

## Support

For issues or questions:
- Check the troubleshooting section above
- Review script output for error messages
- Check GitHub PAT permissions
- Verify Google Drive is mounted

---

## License

These scripts are part of the fl_clean project. Use freely for personal and academic projects.
