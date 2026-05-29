#!/usr/bin/env bash
set -euo pipefail
OUTDIR="${1:-resources/grn_prior}"
mkdir -p "$OUTDIR"

fetch() {
  local url="$1"
  local out="$2"
  echo "[download] $out"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$OUTDIR/$out"
  if head -c 512 "$OUTDIR/$out" | tr 'A-Z' 'a-z' | grep -Eq '<html|<!doctype html|<head|<body'; then
    echo "[error] $out looks like an HTML page, not a TSV. Removing it." >&2
    rm -f "$OUTDIR/$out"
    exit 1
  fi
  if ! head -n 1 "$OUTDIR/$out" | grep -q $'\t'; then
    echo "[warn] $out first line does not contain TABs; please inspect manually." >&2
  fi
}

fetch 'https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1&fields=sources,references,curation_effort,consensus_direction,consensus_stimulation,consensus_inhibition&format=tsv' 'collectri_human.tsv'
fetch 'https://omnipathdb.org/interactions?datasets=tf_target&resources=TRRUST&genesymbols=1&fields=sources,references,curation_effort,consensus_direction,consensus_stimulation,consensus_inhibition&format=tsv' 'trrust_human.tsv'
fetch 'https://omnipathdb.org/interactions?datasets=omnipath&types=post_translational&genesymbols=1&fields=sources,references,curation_effort,consensus_direction,consensus_stimulation,consensus_inhibition&format=tsv' 'omnipath_signaling_human.tsv'
fetch 'https://omnipathdb.org/intercell?genesymbols=1&format=tsv' 'omnipath_intercell_human.tsv'

echo "[ok] files saved in $OUTDIR"
