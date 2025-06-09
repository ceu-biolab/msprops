#!/usr/bin/env python
"""
Compare experimental MS/MS spectra against a reference library using matchms
and report Modified-Cosine similarity scores.

Usage:
    python match_mgf_cosine.py experiment.mgf reference.mgf
           --prec_tol 0.01 --ms2_tol 0.05
"""

import argparse
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, normalize_intensities
from matchms.similarity import ModifiedCosine


def clean(spectrum):
    """Apply default preprocessing recommended by matchms docs."""
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    return spectrum


def load_and_clean(path):
    """Load all spectra from an MGF file and clean them."""
    return [clean(s) for s in load_from_mgf(path)]


def main():
    p = argparse.ArgumentParser(
        description="Compute Modified-Cosine scores between two MGF files")
    p.add_argument("experiment", help="Experimental .mgf (queries)")
    p.add_argument("library", help="Reference .mgf")
    p.add_argument("--prec_tol", type=float, default=0.01,
                   help="Precursor m/z tolerance in Da (default 0.01)")
    p.add_argument("--ms2_tol", type=float, default=0.05,
                   help="MS2 peak tolerance in Da (default 0.05)")
    args = p.parse_args()

    queries = load_and_clean(args.experiment)
    refs = load_and_clean(args.library)
    scorer = ModifiedCosine(tolerance=args.ms2_tol)

    print("query_id\tquery_name\tquery_precursor_mz\treference_id\treference_name\treference_precursor_mz\tprecursor_diff\tmcosine_score\tmatches")
    for q in queries:
        q_prec = q.get("precursor_mz")
        q_id = q.get("scans") or q.get("id") or q.metadata.get("title", "query")
        q_name = q.get("name") or q.metadata.get("name", "")
        for r in refs:
            r_prec = r.get("precursor_mz") or 0
            diff = abs(r_prec - (q_prec or 0))
            if diff <= args.prec_tol:
                result = scorer.pair(q, r)
                r_id = r.get("scans") or r.get("id") or r.metadata.get("title", "reference")
                r_name = r.get("name") or r.metadata.get("name", "")
                print(f"{q_id}\t{q_name}\t{q_prec or 0:.6f}\t{r_id}\t{r_name}\t{r_prec:.6f}\t{diff:.6f}\t"
                      f"{result['score']:.4f}\t{result['matches']}")


if __name__ == "__main__":
    main()
