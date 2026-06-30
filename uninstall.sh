#!/usr/bin/env bash
#
# uninstall.sh — remove meeting-ninja and reclaim its disk footprint.
#
# macOS does NOT auto-clean ~/.cache or ~/.local, so model caches and venvs
# persist until removed by hand. This script removes meeting-ninja's own
# footprint and PROMPTS before touching anything shared with other tools.
#
# It never deletes your recordings. It will offer to delete the regenerable
# audio/ WAVs, but the original .mov/.m4a files are always left alone.
#
# Usage:   bash uninstall.sh
#          bash uninstall.sh --yes      # don't prompt; take every suggested action
#
set -euo pipefail

ALWAYS_YES="${1:-}"
HOME_DIR="$HOME"

confirm() {
    # confirm "question"  -> returns 0 for yes
    if [ "$ALWAYS_YES" = "--yes" ]; then return 0; fi
    read -r -p "$1 [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

human() { # bytes -> human; falls back gracefully
    du -sh "$1" 2>/dev/null | cut -f1
}

echo "meeting-ninja uninstall / cleanup"
echo "=================================="
echo

# 1. The CLI itself (pipx venv) -------------------------------------------------
if command -v pipx >/dev/null 2>&1 && pipx list 2>/dev/null | grep -q "meeting-ninja"; then
    if confirm "Remove the meeting-ninja CLI (pipx uninstall)?"; then
        pipx uninstall meeting-ninja || true
    fi
else
    echo "• meeting-ninja not installed via pipx (skipping)."
fi

# 2. App state + saved settings (incl. the HF token) ----------------------------
APPDIR="$HOME_DIR/.meeting-transcriber"
if [ -d "$APPDIR" ]; then
    if confirm "Delete app state + settings at $APPDIR ($(human "$APPDIR"))? This holds the saved HF token and the file index."; then
        rm -rf "$APPDIR"
    fi
fi

# 3. Regenerable extracted audio (NOT your recordings) --------------------------
# These live in audio/ folders next to recordings. We only remove the audio/
# directories, never the source media.
echo
echo "Looking for regenerable audio/ folders (extracted WAVs)…"
DEFAULT_MEETINGS="$HOME_DIR/Local Documents/Meetings"
SEARCH_ROOT="$DEFAULT_MEETINGS"
if [ ! -d "$SEARCH_ROOT" ]; then
    read -r -p "Meetings folder not found. Path to search for audio/ folders (blank to skip): " SEARCH_ROOT
fi
if [ -n "${SEARCH_ROOT:-}" ] && [ -d "$SEARCH_ROOT" ]; then
    while IFS= read -r d; do
        [ -z "$d" ] && continue
        if confirm "Delete $d ($(human "$d"))? (regenerable; your recordings stay)"; then
            rm -rf "$d"
        fi
    done < <(find "$SEARCH_ROOT" -type d -name audio 2>/dev/null)
fi

# 4. Whisper model cache (shared by any whisper user) ---------------------------
WHISPER="$HOME_DIR/.cache/whisper"
if [ -d "$WHISPER" ]; then
    echo
    echo "Whisper model cache ($(human "$WHISPER")):"
    ls -1sh "$WHISPER" 2>/dev/null | sed 's/^/    /'
    echo "  (Re-downloaded automatically the next time you use that model.)"
    if confirm "Delete the whole Whisper model cache?"; then
        rm -rf "$WHISPER"
    fi
fi

# 5. pyannote models in the HF cache (only ours) --------------------------------
# IMPORTANT: ~/.cache/huggingface is shared. Only remove pyannote/* — leave any
# other models (e.g. rerankers from other projects) untouched.
HF="$HOME_DIR/.cache/huggingface"
if [ -d "$HF" ]; then
    echo
    echo "Removing only pyannote/* from the shared HuggingFace cache (other models kept)…"
    while IFS= read -r d; do
        [ -z "$d" ] && continue
        if confirm "Delete $d ($(human "$d"))?"; then
            rm -rf "$d"
        fi
    done < <(find "$HF" -maxdepth 3 -type d -iname "*pyannote*" 2>/dev/null)
    echo "  Note: anything non-pyannote in $HF was left alone."
fi

# 6. Project virtualenv (only needed for the old Streamlit UI) ------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
if [ -d "$VENV" ]; then
    echo
    if confirm "Delete the project virtualenv at $VENV ($(human "$VENV"))? Only needed to run the old Streamlit UI."; then
        rm -rf "$VENV"
    fi
fi

echo
echo "Done. Your source recordings were not touched."
echo "If you want the project source gone too, delete this repo folder manually."
