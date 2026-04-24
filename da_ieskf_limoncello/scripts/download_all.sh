#!/usr/bin/env bash
# download_all.sh — download all evaluation datasets
# MCD (ntu_day sequences), GrandTour, City02, R-Campus
# Run on monster; downloads go to ~/datasets/
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-$HOME/datasets}"
mkdir -p "$DATASETS_ROOT"

echo "=== Downloading datasets to $DATASETS_ROOT ==="
echo "Date: $(date -u)"
echo ""

# ── Helper ───────────────────────────────────────────────────────────────────
download_if_missing() {
    local name="$1"
    local url="$2"
    local dest="$3"
    if [ -d "$dest" ] || [ -f "$dest" ]; then
        echo "[$name] Already exists — skipping."
        return 0
    fi
    echo "[$name] Downloading from $url ..."
    mkdir -p "$(dirname "$dest")"
    case "$url" in
        *.zip)
            wget -q --show-progress -O "${dest}.zip" "$url"
            unzip -q "${dest}.zip" -d "$(dirname "$dest")"
            rm "${dest}.zip"
            ;;
        *.tar.gz|*.tgz)
            wget -q --show-progress -O "${dest}.tar.gz" "$url"
            tar -xzf "${dest}.tar.gz" -C "$(dirname "$dest")"
            rm "${dest}.tar.gz"
            ;;
        *.bag|*.db3)
            wget -q --show-progress -O "$dest" "$url"
            ;;
        *)
            wget -q --show-progress -O "$dest" "$url"
            ;;
    esac
    echo "[$name] Done."
}

# ── MCD (Multiple Camera + LiDAR Dataset) ────────────────────────────────────
# https://mcdviral.github.io/
MCD_ROOT="$DATASETS_ROOT/mcd"
mkdir -p "$MCD_ROOT"

echo "--- MCD dataset ---"
echo "MCD sequences need manual download from https://mcdviral.github.io/"
echo "Register and download these sequences to $MCD_ROOT:"
echo "  ntu_day_01   (revisits — loop closure benchmark)"
echo "  ntu_day_02   (shortest, target APE 0.156 m — Phase 1 gate)"
echo "  ntu_day_10   (long sequence, outdoor)"
echo ""
echo "After download, place bags as:"
echo "  $MCD_ROOT/ntu_day_01/ntu_day_01.bag"
echo "  $MCD_ROOT/ntu_day_01/gt.tum"
echo "  $MCD_ROOT/ntu_day_02/ntu_day_02.bag"
echo "  $MCD_ROOT/ntu_day_02/gt.tum"
echo ""

# Automated: download ground truth TUM files if public links exist
# (Adjust URLs once confirmed from the MCD release page)
NTU_GT_BASE="https://mcdviral.github.io/static/ground_truth"
for seq in ntu_day_01 ntu_day_02 ntu_day_10; do
    GT_PATH="$MCD_ROOT/$seq/gt.tum"
    if [ ! -f "$GT_PATH" ]; then
        mkdir -p "$MCD_ROOT/$seq"
        echo "[$seq] Attempting GT download (may fail if URL changed)..."
        wget -q --show-progress \
            -O "$GT_PATH" \
            "$NTU_GT_BASE/${seq}_gt.tum" 2>/dev/null \
            || echo "[$seq] GT download failed — download manually."
    fi
done

# ── HILTI GrandTour ──────────────────────────────────────────────────────────
# https://hilti-challenge.com/dataset-2022.html
GRAND_ROOT="$DATASETS_ROOT/grand_tour"
mkdir -p "$GRAND_ROOT"

echo "--- HILTI GrandTour dataset ---"
echo "Download from https://hilti-challenge.com/dataset-2022.html"
echo "Target sequence: grand_tour (long indoor+outdoor traversal)"
echo "Place bag at: $GRAND_ROOT/grand_tour.bag"
echo ""

# ── City02 (Hong Kong City) ──────────────────────────────────────────────────
# https://www.ram-lab.com/file/dataset/city02/
CITY02_ROOT="$DATASETS_ROOT/city02"
mkdir -p "$CITY02_ROOT"

echo "--- City02 dataset ---"
echo "Download from https://www.ram-lab.com/file/dataset/city02/"
echo "This is the PRIMARY degenerate environment (urban canyons/tunnels)."
echo "Place bag at: $CITY02_ROOT/city02.bag"
echo ""

# If public download URL is available:
# download_if_missing "city02" "https://..." "$CITY02_ROOT/city02.bag"

# ── R-Campus (Revisit Campus) ─────────────────────────────────────────────────
# Part of MCD or separate depending on release
RCAMPUS_ROOT="$DATASETS_ROOT/r_campus"
mkdir -p "$RCAMPUS_ROOT"

echo "--- R-Campus dataset ---"
echo "Download from https://mcdviral.github.io/ (campus sequences)"
echo "Place bag at: $RCAMPUS_ROOT/r_campus.bag"
echo ""

# ── Verify ────────────────────────────────────────────────────────────────────
echo "=== Download summary ==="
for path in \
    "$MCD_ROOT/ntu_day_01" \
    "$MCD_ROOT/ntu_day_02" \
    "$GRAND_ROOT" \
    "$CITY02_ROOT" \
    "$RCAMPUS_ROOT"; do
    if [ -d "$path" ]; then
        SIZE=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  [OK]     $path  ($SIZE)"
    else
        echo "  [MISSING] $path"
    fi
done

echo ""
echo "Run evaluate_all.sh after all datasets are in place."
