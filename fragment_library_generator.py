#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

import pymzml
import yaml
import numpy as np
import pyopenms as oms
from matchms import Fragments
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from matchms.filtering import (
    default_filters,
    select_by_mz,
    select_by_relative_intensity,
    reduce_to_number_of_peaks,
    require_minimum_number_of_peaks,
    remove_peaks_around_precursor_mz,
    remove_peaks_outside_top_k,
)


def openms_deisotope(s: Spectrum, tol_ppm=10):
    spec = _matchms_to_msspectrum(s)
    oms.Deisotoper.deisotopeAndSingleChargeDefault(spec, fragment_tolerance=tol_ppm, fragment_unit_ppm=True)
    spec.sortByPosition()
    s = _msspectrum_to_matchms(spec)
    if len(s.peaks.mz) == 0:
        logging.debug("Deisotoping removed all peaks from spectrum %s", s.metadata.get("scan_id", "unknown"))
        return None
    return s

# ------------------------------------------------------------
# Registry – add new filters here when you test them
# ------------------------------------------------------------
FILTER_REGISTRY = {
    "default_filters": default_filters,
    "select_by_mz": select_by_mz,
    "select_by_relative_intensity": select_by_relative_intensity,
    "reduce_to_number_of_peaks": reduce_to_number_of_peaks,
    "require_minimum_number_of_peaks": require_minimum_number_of_peaks,
    "remove_peaks_around_precursor_mz": remove_peaks_around_precursor_mz,
    "remove_peaks_outside_top_k": remove_peaks_outside_top_k,
    "openms_deisotope": openms_deisotope,
}

SpectrumFilter = Union[str, Dict[str, Any]]


def read_config(path: Path) -> dict:
    """Read YAML config into dict."""
    with open(path, "rt", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ppm_diff(a: float, b: float, target_mz: float) -> float:
    """Absolute ppm difference."""
    return abs(a - b) / target_mz * 1e6


def build_processor(filter_chain: List[SpectrumFilter]):
    """Return a function that applies the configured filters in sequence."""

    def apply_filter(spec: Spectrum, entry: SpectrumFilter):
        if isinstance(entry, str):
            name, kwargs = entry, {}
        elif isinstance(entry, dict):
            name = entry.get("name")
            kwargs = entry.get("args", {})
        else:
            logging.error("Unhandled filter entry: %s", entry)
            return spec
        fn = FILTER_REGISTRY.get(name)
        if fn is None:
            logging.warning("Unknown filter '%s' skipped (update FILTER_REGISTRY)", name)
            return spec
        return fn(spec, **kwargs) if kwargs else fn(spec)

    def processor(spec: Spectrum) -> Spectrum:
        for entry in filter_chain:
            spec = apply_filter(spec, entry)
            if spec is None:
                break  # some filters may return None to drop spectrum
        return spec

    return processor


def _matchms_to_msspectrum(ms: Spectrum) -> "oms.MSSpectrum":
    """Convert a matchms Spectrum into an OpenMS MSSpectrum (MS2)."""
    spec = oms.MSSpectrum()
    spec.set_peaks((ms.peaks.mz.astype(float, copy=False),
                    ms.peaks.intensities.astype(float, copy=False)))
    spec.sortByPosition()
    spec.setMSLevel(2)
    # propagate RT (s) if present – OpenMS stores RT in minutes internally
    if "rt" in ms.metadata:
        spec.setRT(float(ms.metadata["rt"]))
    # set precursor information if available
    prec = oms.Precursor()
    prec_mz = ms.metadata.get("precursor_mz") or ms.metadata.get("pepmass")
    if prec_mz is not None:
        prec.setMZ(float(prec_mz))
    charge = ms.metadata.get("charge")
    if charge is not None:
        prec.setCharge(int(charge))
    spec.setPrecursors([prec])
    # use scan‐id/native‐id if present (preserves traceability)
    if "scan_id" in ms.metadata:
        spec.setNativeID(str(ms.metadata["scan_id"]))
    return spec


def _msspectrum_to_matchms(ms: "oms.MSSpectrum") -> Spectrum:
    """Convert an OpenMS MSSpectrum back to matchms Spectrum."""
    mz, inten = ms.get_peaks()
    meta = {
        "scan_id": ms.getNativeID(),
        "rt": ms.getRT(),
        "num_scans": len(ms.getNativeID().split(",")),
    }
    if ms.getPrecursors():
        prec = ms.getPrecursors()[0]
        meta["precursor_mz"] = prec.getMZ()
        meta["charge"] = prec.getCharge()
    return Spectrum(mz=np.asarray(mz, dtype=float),
                    intensities=np.asarray(inten, dtype=float),
                    metadata=meta)


def merge_spectra(spectra: List[Spectrum], cfg: dict) -> Optional[Spectrum]:
    if not spectra:
        return None

    mcfg = cfg.get("merge", {})

    exp = oms.MSExperiment()
    for sp in spectra:
        exp.addSpectrum(_matchms_to_msspectrum(sp))
    target_mz = cfg.get("precursor_mz", spectra[0].metadata.get("precursor_mz"))

    merger = oms.SpectraMerger()
    params = merger.getParameters()
    
    ppm = mcfg.get("mz_bin_ppm")
    da = mcfg.get("mz_bin_da")

    if not ppm and not da:
        logging.error("No m/z tolerance specified in config: 'mz_bin_ppm' or 'mz_bin_da' required")
        return None
    if ppm and da:
        logging.warning("Both 'mz_bin_ppm' and 'mz_bin_da' specified, using 'mz_bin_ppm'")

    if ppm:
        params.setValue("mz_binning_width", float(ppm))
        params.setValue("mz_binning_width_unit", "ppm")
    else:
        params.setValue("mz_binning_width", float(da))
        params.setValue("mz_binning_width_unit", "Da")

    if mcfg.get("rt_prec_tolerance"):
        rt_tol = mcfg["rt_prec_tolerance_s"]
        params.setValue("precursor_method:rt_tolerance", float(rt_tol))

    params.setValue("precursor_method", "RT")

    params.setValue("sort_blocks", "RT_ascending")

    merger.setParameters(params)

    merger.mergeSpectraPrecursors(exp)

    merged = None
    max_cluster = 0
    da_tol_prec = get_mz_tol_da(cfg, target_mz)
    for ms in exp.getSpectra():
        precursors = ms.getPrecursors()
        if not precursors or abs(precursors[0].getMZ() - target_mz) > da_tol_prec:
            continue

        # In the rare case of 2+ spectra for same precursor, pick the one with most scans
        cluster_size = 1 + ms.getNativeID().count(",")
        if cluster_size > max_cluster:
            max_cluster, merged = cluster_size, ms

    if merged is not None:
        logging.debug("Merged %d spectra via OpenMS SpectraMerger", max_cluster)
        return _msspectrum_to_matchms(merged)

    else:
        logging.warning("No merged spectrum found for precursor m/z %.6f", target_mz)
        return None

def get_mz_tol_da(cfg, target_mz):
    mz_tol_ppm = cfg.get("mz_tol_ppm", None)
    mz_tol_da = cfg.get("mz_tol_da", None)
    if mz_tol_ppm is not None:
        da_tol = target_mz * mz_tol_ppm / 1e6
    elif mz_tol_da is not None:
        da_tol = mz_tol_da
    else:
        logging.error("No m/z tolerance specified in config: 'mz_tol_ppm' or 'mz_tol_da' required")
        raise ValueError("No m/z tolerance specified in config")
    return da_tol


def merge_spectra_apex(spectra: List[Spectrum], cfg: dict) -> Optional[Spectrum]:
    """
    Simplest merging of spectra by picking the tallest data point within a mass tolerance.
    """
    if not spectra:
        return None
    if len(spectra) == 1:
        return spectra[0]

    mcfg = cfg.get("merge", {})
    tol_ppm = mcfg.get("mz_bin_ppm")
    tol_da = mcfg.get("mz_bin_da")
    if tol_ppm is None and tol_da is None:
        logging.error("No m/z tolerance specified in config: 'mz_bin_ppm' or 'mz_bin_da' required")
        return None

    # Add some margin to account for lack of dynamic centroiding
    if tol_ppm is not None:
        tol_ppm *= 1.5
    if tol_da is not None:
        tol_da *= 1.5

    ### Step 1: Get all data points as (m/z, intensity) pairs

    all_peaks = []
    for spec in spectra:
        mz, intens = spec.peaks.mz, spec.peaks.intensities
        if mz.size == 0:
            continue
        all_peaks.extend(zip(mz, intens))
    if not all_peaks:
        logging.warning("No peaks found in input spectra for merging")
        return None
    all_peaks = np.array(all_peaks, dtype=float)
    # Order by intensity descending
    all_peaks = all_peaks[np.argsort(-all_peaks[:, 1])]

    ### Step 2: Group peaks by m/z within the specified tolerance

    merged_peaks = []
    picked = set()
    for i,t in enumerate(all_peaks):
        if t[0] in picked:
            continue

        # Pick the largest non picked peak
        merged_peaks.append((t[0], t[1]))

        # Mark all within mass tolerance as picked
        for other in all_peaks[i+1:]:
            if tol_da is not None:
                if abs(other[0] - t[0]) <= tol_da:
                    picked.add(other[0])
            elif tol_ppm is not None:
                if ppm_diff(other[0], t[0], t[0]) <= tol_ppm:
                    picked.add(other[0])
            else:
                logging.error("No m/z tolerance specified in config: 'mz_bin_ppm' or 'mz_bin_da' required")
                raise ValueError("No m/z tolerance specified in config")

    if not merged_peaks:
        logging.warning("No peaks merged from input spectra")
        return None

    ### Step 3: Create a new sorted Spectrum from the merged peaks

    mz, intens = zip(*merged_peaks)
    mz = np.array(mz, dtype=float)
    intens = np.array(intens, dtype=float)
    order = np.argsort(mz)  # ascending indices
    mz, intens = mz[order], intens[order]

    merged_spec = Spectrum(
        mz=mz.astype(float, copy=False),
        intensities=intens.astype(float, copy=False),
        metadata={
            "precursor_mz": cfg.get("precursor_mz", spectra[0].metadata.get("precursor_mz")),
            "source_file": spectra[0].metadata.get("source_file"),
            "charge": spectra[0].metadata.get("charge"),
        }
    )
    merged_spec.peaks = Fragments(
        mz=mz.astype(float, copy=False),
        intensities=intens.astype(float, copy=False),
    )
    logging.debug("Merged %d spectra into one with %d peaks", len(spectra), len(merged_peaks))

    return merged_spec


def consolidate_spectrum(spec: Spectrum, cfg: dict) -> Spectrum:
    """Final clean-up: thresholding, top-N, scaling."""
    ccfg = cfg.get("consolidate", {})
    if spec is None:
        return None

    mz, intens = spec.peaks.mz, spec.peaks.intensities
    if intens.size == 0:
        return spec

    # 4.1 relative intensity floor
    floor = ccfg.get("min_relative_intensity", 0.0)
    if floor > 0:
        base = intens.max()
        keep = intens >= base * floor
        mz, intens = mz[keep], intens[keep]

    # 4.2 keep top-N
    top_n = ccfg.get("top_n")
    if top_n is not None and intens.size > top_n:
        idx = np.argsort(intens)[-top_n:]
        mz, intens = mz[idx], intens[idx]

    # 4.3 scale so base = intensity_norm
    norm_to = ccfg.get("intensity_norm", 1000)
    if intens.max() > 0:
        intens = intens * (norm_to / intens.max())

    order = np.argsort(mz)          # ascending indices
    mz, intens = mz[order], intens[order]

    from matchms import Fragments    # 1-time import at top of file
    spec.peaks = Fragments(
        mz=mz.astype(float, copy=False),
        intensities=intens.astype(float, copy=False),
    )
    return spec


def extract_ms2_spectra_from_mzml(mzml_path: Path, processor, target_mz=None, mz_tol_ppm=None, rt_window=None, expected_charge=None):
    """
    Extract MS2 spectra from an mzML file.
    
    If target_mz is specified, only extracts spectra matching that precursor m/z.
    Otherwise, extracts all MS2 spectra.
    
    Returns a dictionary mapping precursor m/z (rounded to 4 decimals) to a list of spectra.
    """
    reader = pymzml.run.Reader(
        mzml_path,
        extraAccessions=["MS:1000744"],  # selected ion m/z
    )
    
    spectra_by_precursor = {}
    
    for scan in reader:
        if scan.ms_level != 2:
            continue
        if not scan.selected_precursors:
            continue
            
        try:
            prec_mz = scan.selected_precursors[0]["mz"] if scan.selected_precursors else scan["precursorMz"]
        except Exception:
            logging.warning("No precursor m/z found in scan %s", scan.ID)
            continue
            
        # Check if we're filtering by expected charge
        if expected_charge is not None and scan.selected_precursors[0].get("charge") != expected_charge:
            logging.warning("Scan %s has unexpected charge %s (expected %d)", 
                           scan.ID, scan.selected_precursors[0].get("charge"), expected_charge)
            continue
            
        # Check if we're filtering by target m/z
        if target_mz is not None and mz_tol_ppm is not None:
            if ppm_diff(prec_mz, target_mz, target_mz) > mz_tol_ppm:
                continue
                
        # Check RT window if specified
        rt = scan.scan_time_in_minutes() * 60.0
        if rt_window is not None and not (rt_window[0] <= rt <= rt_window[1]):
            continue
            
        spec = Spectrum(
            mz=scan.mz,
            intensities=scan.i,
            metadata={
                "precursor_mz": prec_mz,
                "rt": rt,
                "scan_id": scan.ID,
                "source_file": mzml_path.name,
                "charge": scan.selected_precursors[0].get("charge"),
                "collision_energy": scan.selected_precursors[0].get("collisionEnergy"),
            },
        )
        
        spec = processor(spec)
        if spec is not None and len(spec.peaks.mz) >= 3:
            # Round to 4 decimal places to group similar precursors
            prec_key = round(prec_mz, 4)
            if prec_key not in spectra_by_precursor:
                spectra_by_precursor[prec_key] = []
            spectra_by_precursor[prec_key].append(spec)
            
    return spectra_by_precursor

def extract_ms2_spectra_from_multiple_mzml(mzml_paths: List[Path], processor, target_mz=None, mz_tol_ppm=None, rt_window=None, expected_charge=None):
    """
    Extract MS2 spectra from multiple mzML files and combine the results.
    
    Returns a dictionary mapping precursor m/z (rounded to 4 decimals) to a list of spectra.
    """
    spectra_by_precursor = {}
    
    for mzml_path in mzml_paths:
        logging.info("Extracting spectra from %s", mzml_path)
        file_spectra = extract_ms2_spectra_from_mzml(
            mzml_path, processor, target_mz, mz_tol_ppm, rt_window, expected_charge
        )
        
        # Merge the spectra dictionaries
        for prec_mz, spectra in file_spectra.items():
            if prec_mz not in spectra_by_precursor:
                spectra_by_precursor[prec_mz] = []
            spectra_by_precursor[prec_mz].extend(spectra)
    
    # Log some statistics
    total_spectra = sum(len(spectra) for spectra in spectra_by_precursor.values())
    logging.info("Extracted %d total spectra across %d precursors from %d files", 
                 total_spectra, len(spectra_by_precursor), len(mzml_paths))
    
    return spectra_by_precursor

def process_precursor_spectra(spectra: List[Spectrum], precursor_mz: float, cfg: dict, mzml_paths: Union[Path, List[Path]]) -> Optional[Spectrum]:
    """Process all spectra for a given precursor m/z into a single consensus spectrum."""
    if not spectra:
        return None
        
    # Create a copy of the config with this specific precursor m/z
    prec_cfg = dict(cfg)
    prec_cfg["precursor_mz"] = precursor_mz
    
    # Merge spectra for this precursor
    if cfg.get("merge", {}).get("method") == "weighted":
        consensus = merge_spectra(spectra, prec_cfg)
    elif cfg.get("merge", {}).get("method") in ["simple", "apex"]:
        consensus = merge_spectra_apex(spectra, prec_cfg)
    else:
        logging.warning("No valid merge method specified, using 'simple' as default")
        consensus = merge_spectra_apex(spectra, prec_cfg)
    
    consensus = consolidate_spectrum(consensus, prec_cfg)
    if consensus is not None:
        # Add some extra metadata
        if isinstance(mzml_paths, list) and len(mzml_paths) > 1:
            # For multiple files, create a combined name
            base_name = Path(mzml_paths[0].stem).stem  # Remove potential nested extensions
            consensus.metadata["name"] = f"combined_{base_name}_mz{precursor_mz:.4f}"
            # Add list of source files
            consensus.metadata["source_files"] = [p.name for p in mzml_paths]
        else:
            # Single file case
            mzml_path = mzml_paths[0] if isinstance(mzml_paths, list) else mzml_paths
            consensus.metadata["name"] = f"{mzml_path.stem}_mz{precursor_mz:.4f}"
        
        # Number of fragments
        consensus.metadata["num_fragments"] = len(consensus.peaks.mz)
    
    return consensus

def extract_and_filter_spectra(mzml_paths: List[Path], cfg: dict, target_mz=None, expected_charge=None, all_precursors=False):
    """
    Extract and filter MS2 spectra from one or more mzML files.
    
    Returns a dictionary mapping precursor m/z to lists of filtered spectra.
    """
    processor = build_processor(cfg.get("filters", []))
    
    if all_precursors:
        file_desc = f"{len(mzml_paths)} file{'s' if len(mzml_paths) > 1 else ''}"
        logging.info(f"Extracting all precursors from {file_desc}")
    else:
        target_mz = cfg["precursor_mz"] if target_mz is None else target_mz
        mz_tol_ppm = cfg["mz_tol_ppm"]
        rt_window = cfg["rt_window"]
        logging.info("Precursor %.6f ± %.1f ppm | RT %.1f-%.1f s from %d file%s", 
                    target_mz, mz_tol_ppm, rt_window[0], rt_window[1],
                    len(mzml_paths), "s" if len(mzml_paths) > 1 else "")
    
    # Extract and filter MS2 spectra from all mzML files
    spectra_by_precursor = extract_ms2_spectra_from_multiple_mzml(
        mzml_paths,
        processor,
        None if all_precursors else target_mz,
        None if all_precursors else cfg["mz_tol_ppm"],
        None if all_precursors else cfg["rt_window"],
        expected_charge
    )
    
    if not spectra_by_precursor:
        logging.warning("No matching spectra found in input files")
    
    return spectra_by_precursor

def merge_and_consolidate_spectra(spectra_by_precursor: Dict[float, List[Spectrum]], cfg: dict, mzml_paths: List[Path], target_mz=None, all_precursors=False):
    """
    Merge and consolidate spectra for each precursor.
    
    Returns a list of consensus spectra.
    """
    if not spectra_by_precursor:
        return []
        
    output_spectra = []
    
    if all_precursors:
        # Process all precursors found
        for prec_mz, spectra in spectra_by_precursor.items():
            consensus = process_precursor_spectra(spectra, prec_mz, cfg, mzml_paths)
            if consensus is not None:
                output_spectra.append(consensus)
    else:
        # Process just the target precursor
        target_key = round(target_mz, 4)
        if target_key not in spectra_by_precursor:
            logging.warning("No spectra found for precursor m/z %.6f", target_mz)
            return []
            
        consensus = process_precursor_spectra(spectra_by_precursor[target_key], target_mz, cfg, mzml_paths)
        if consensus is not None:
            output_spectra.append(consensus)
    
    return output_spectra

def parse_args():
    p = argparse.ArgumentParser(description="Extract + clean MS² scans for a precursor.")
    p.add_argument("--mzml", required=True, type=Path, nargs='+', help="Input mzML file(s)")
    p.add_argument("--config", required=True, type=Path, help="YAML config")
    p.add_argument("--output", required=True, type=Path, help="Output MGF path")
    p.add_argument("--precursor_mz", type=float, help="Override precursor_mz")
    p.add_argument("--mz_tol_ppm", type=float, help="Override m/z tolerance (ppm)")
    p.add_argument("--rt_window", nargs=2, type=float, metavar=("START", "END"), help="RT window in seconds")
    p.add_argument("--charge", type=int, help="Expected precursor charge state")
    p.add_argument("--all_precursors", action="store_true", help="Extract all precursors instead of the one specified")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.loglevel, format="%(levelname)s: %(message)s")

    cfg = read_config(args.config)
    if args.precursor_mz is not None:
        cfg["precursor_mz"] = args.precursor_mz
    if args.mz_tol_ppm is not None:
        cfg["mz_tol_ppm"] = args.mz_tol_ppm
    if args.rt_window is not None:
        cfg["rt_window"] = list(args.rt_window)
    if args.charge is not None:
        cfg["charge"] = args.charge

    # Auto-append .mgf extension if missing
    output_path = args.output if args.output.suffix else args.output.with_suffix(".mgf")
    
    # Step 1: Extract and filter spectra from mzML files
    filtered_spectra = extract_and_filter_spectra(
        args.mzml,
        cfg,
        args.precursor_mz,
        args.charge,
        args.all_precursors
    )
    
    # Step 2: Merge and consolidate spectra by precursor
    consensus_spectra = merge_and_consolidate_spectra(
        filtered_spectra,
        cfg,
        args.mzml,
        args.precursor_mz,
        args.all_precursors
    )
    
    # Step 3: Save output
    if consensus_spectra:
        save_as_mgf(consensus_spectra, str(output_path))
        logging.info("Saved %d spectrum%s ➜ %s",
                    len(consensus_spectra), "" if len(consensus_spectra)==1 else "s", output_path)
    else:
        logging.warning("No spectra to save")


if __name__ == "__main__":
    main()
