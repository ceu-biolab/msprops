#!/usr/bin/env python3
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from matchms import set_matchms_logger_level
from matchms.importing import load_from_mzml
from matchms.importing.parsing_utils import parse_mzml_mzxml_metadata
from collections import Counter, defaultdict

# 1) Suppress MatchMS warnings
set_matchms_logger_level("ERROR")

def is_profile_mode(spectrum, cv_thresh=0.01):
    """Detect profile vs centroid mode by m/z spacing variation."""
    mz = spectrum.peaks.mz
    if mz.size < 3:
        return False
    diffs = np.diff(mz)
    cv = np.std(diffs) / np.mean(diffs)
    return cv < cv_thresh

def get_experiment_metadata(mzml_file):
    """Extract important experiment metadata like ionization technique and polarity directly from the XML."""
    # Initialize return values
    ionization = "Unknown"
    polarity = "Unknown"
    instrument = "Unknown"
    manufacturer = "Unknown"
    
    try:
        # Parse the mzML file directly with ElementTree for better navigation
        namespaces = {
            'ms': 'http://psi.hupo.org/ms/mzml',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
        
        # First try quick targeted parsing for the elements we need
        with open(mzml_file, 'r') as f:
            # Read the first part of the file where instrument info is usually found
            header = f.read(50000)  # Read a large enough chunk to contain the header
            
            # Check for ionization technique
            ionization_match = None
            for ion_type in ["MS:1000073", "MS:1000074", "MS:1000075", "electrospray ionization", 
                            "atmospheric pressure chemical ionization", "maldi"]:
                if ion_type in header:
                    ionization_match = ion_type
                    break
            
            if ionization_match:
                ionization = ionization_match
            
            # Check for polarity information
            if "MS:1000130" in header:
                polarity = "Positive"
            elif "MS:1000129" in header:
                polarity = "Negative"
        
        # If quick parsing didn't work, do more thorough XML parsing
        if ionization == "Unknown" or polarity == "Unknown":
            tree = ET.parse(mzml_file)
            root = tree.getroot()
            
            # Strip namespace if present
            if '}' in root.tag:
                ns = root.tag.split('}')[0] + '}'
            else:
                ns = ''
            
            # Look for instrument configuration
            instrument_config = root.find(f".//{ns}instrumentConfigurationList/{ns}instrumentConfiguration")
            if instrument_config is not None:
                # Get instrument model/manufacturer
                for cvParam in instrument_config.findall(f".//{ns}cvParam"):
                    if "model" in cvParam.get("name", "").lower():
                        instrument = cvParam.get("name", "Unknown")
                    elif "manufacturer" in cvParam.get("name", "").lower():
                        manufacturer = cvParam.get("name", "Unknown")
                
                # Find source information for ionization
                source = instrument_config.find(f".//{ns}componentList/{ns}source")
                if source is not None:
                    for cvParam in source.findall(f".//{ns}cvParam"):
                        param_name = cvParam.get("name", "")
                        param_accession = cvParam.get("accession", "")
                        if any(ion_term in param_name.lower() for ion_term in 
                              ["ionization", "electrospray", "esi", "maldi", "apci"]):
                            ionization = f"{param_name} ({param_accession})"
                            break
            
            # Look for polarity in scan settings
            for scan_settings in root.findall(f".//{ns}scanSettingsList/{ns}scanSettings"):
                for cvParam in scan_settings.findall(f".//{ns}cvParam"):
                    accession = cvParam.get("accession", "")
                    if accession == "MS:1000130":
                        polarity = "Positive"
                        break
                    elif accession == "MS:1000129":
                        polarity = "Negative"
                        break
            
            # Also check for polarity directly in spectrum elements as shown in the example
            if polarity == "Unknown":
                for spectrum in root.findall(f".//{ns}spectrum"):
                    for cvParam in spectrum.findall(f".//{ns}cvParam"):
                        accession = cvParam.get("accession", "")
                        if accession == "MS:1000130":
                            polarity = "Positive (from spectrum)"
                            break
                        elif accession == "MS:1000129":
                            polarity = "Negative (from spectrum)"
                            break
                    if polarity != "Unknown":
                        break  # Stop checking if we've found polarity
    
    except Exception as e:
        print(f"Warning: Error parsing mzML file directly: {e}")
    
    return {
        "ionization_technique": ionization,
        "polarity": polarity,
        "instrument": instrument,
        "manufacturer": manufacturer
    }

def count_polarities_in_spectra(spectra):
    """Count positive and negative scan polarities in the spectra."""
    positive_count = 0
    negative_count = 0
    unknown_count = 0
    
    for spectrum in spectra:
        meta = spectrum.metadata
        if meta.get("positive_scan") is True:
            positive_count += 1
        elif meta.get("negative_scan") is True:
            negative_count += 1
        else:
            unknown_count += 1
    
    return {
        "positive": positive_count,
        "negative": negative_count,
        "unknown": unknown_count
    }

def count_collision_energies(spectra):
    """Count and categorize spectra by collision energy, looking in the correct activation path."""
    # Initialize counter
    ce_counts = Counter()
    
    # First try to extract collision energy directly from spectra objects
    for spec in spectra:
        meta = spec.metadata
        ce_value = None
        
        # First try to get collision energy from precursor activation info
        # This is the correct path in mzML: spectrum > precursorList > precursor > activation > cvParam
        if "precursor_activation" in meta:
            # Extract from precursor_activation if available
            activation = meta.get("precursor_activation", {})
            if isinstance(activation, dict):
                for param_name, param_value in activation.items():
                    if "collision energy" in param_name.lower():
                        ce_value = param_value
                        break
        
        # Second, look in all params arrays (generic approach)
        if ce_value is None:
            for param_list_name in ["precursor_info", "params"]:
                param_list = meta.get(param_list_name, [])
                if not isinstance(param_list, list):
                    continue
                
                for param in param_list:
                    if isinstance(param, dict):
                        # Check for collision energy by accession or name
                        if param.get("accession") == "MS:1000045" or "collision energy" in param.get("name", "").lower():
                            ce_value = param.get("value", "")
                            break
                
                if ce_value is not None:
                    break
        
        # Last resort: direct metadata field
        if ce_value is None:
            ce_value = meta.get("collision_energy")
        
        # Format the CE value for display
        if ce_value is None:
            ce_str = "Not specified"
        elif isinstance(ce_value, (int, float)):
            ce_str = f"{ce_value:.1f} eV"
        else:
            # Try to convert string value to float if possible
            try:
                ce_float = float(ce_value)
                ce_str = f"{ce_float:.1f} eV"
            except (ValueError, TypeError):
                ce_str = str(ce_value)
        
        ce_counts[ce_str] += 1
    
    # If we couldn't find collision energy in any spectrum, try direct XML parsing
    if len(ce_counts) <= 1 and "Not specified" in ce_counts:
        # Get the first spectrum's metadata to extract the source file path
        if spectra and hasattr(spectra[0], "metadata") and "source_file" in spectra[0].metadata:
            mzml_file = spectra[0].metadata.get("source_file")
            
            try:
                tree = ET.parse(mzml_file)
                root = tree.getroot()
                
                # Strip namespace if present
                if '}' in root.tag:
                    ns = root.tag.split('}')[0] + '}'
                else:
                    ns = ''
                
                # Follow the exact path: spectrum > precursorList > precursor > activation > cvParam
                ce_values = []
                for spectrum in root.findall(f".//{ns}spectrum"):
                    # Skip MS1 spectra (no precursors)
                    ms_level_elem = spectrum.find(f".//{ns}cvParam[@accession='MS:1000511']")
                    if ms_level_elem is None or ms_level_elem.get("value") != "2":
                        continue
                        
                    precursor_list = spectrum.find(f".//{ns}precursorList")
                    if precursor_list is None:
                        continue
                        
                    for precursor in precursor_list.findall(f".//{ns}precursor"):
                        activation = precursor.find(f".//{ns}activation")
                        if activation is None:
                            continue
                            
                        # Look for collision energy parameter
                        ce_param = activation.find(f".//{ns}cvParam[@accession='MS:1000045']")
                        if ce_param is not None:
                            ce_value = ce_param.get("value")
                            if ce_value:
                                try:
                                    ce_float = float(ce_value)
                                    ce_values.append(f"{ce_float:.1f} eV")
                                except (ValueError, TypeError):
                                    ce_values.append(f"{ce_value} eV")
                
                # Count the values found via XML parsing
                if ce_values:
                    ce_counts = Counter(ce_values)
                    
            except Exception as e:
                print(f"Warning: Error parsing mzML file for collision energy: {e}")
    
    return ce_counts

def summarize_spectra(spectra, ms_level):
    n = len(spectra)
    print(f"MS{ms_level} spectra count: {n}")
    
    if n == 0:
        return
        
    # Profile-mode count
    prof = sum(is_profile_mode(s) for s in spectra)
    print(f"  Profile-mode: {prof}/{n}")
    # m/z range and average peak count
    all_mz = np.hstack([s.peaks.mz for s in spectra])
    peak_counts = [s.peaks.mz.size for s in spectra]
    print(f"  m/z range: {all_mz.min():.4f} - {all_mz.max():.4f}")
    print(f"  Peaks per spec: {np.mean(peak_counts):.1f} ± {np.std(peak_counts):.1f}")
    
    # Show full details for MS2, but not metadata keys for MS1
    if ms_level == 2:
        for i, spec in enumerate(spectra[:5], start=1):
            meta = spec.metadata
            print(f"  #{i}: precursor_mz={meta.get('precursor_mz')} | "
                  f"charge={meta.get('charge')} | RT={meta.get('retentionTime')} | "
                  f"profile={is_profile_mode(spec)}")

def ms2_precursor_metrics(spectra):
    """Aggregate and report metrics per valid MS2 precursor."""
    precursors = {}
    for spec in spectra:
        pmz = spec.metadata.get("precursor_mz")
        if pmz is None:
            continue
        key = round(pmz, 4)
        precursors.setdefault(key, []).append(spec)
    
    # Create sorted list of (pmz, count, avg_peaks, avg_intensity)
    precursor_data = []
    for pmz, specs in precursors.items():
        count = len(specs)
        avg_peaks = np.mean([s.peaks.mz.size for s in specs])
        avg_intensity = np.mean([s.peaks.intensities.sum() for s in specs])
        precursor_data.append((pmz, count, avg_peaks, avg_intensity))
    
    # Sort by count (descending), then by intensity (descending)
    precursor_data.sort(key=lambda x: (-x[1], -x[3]))
    
    # Display with limit
    print(f"Unique MS2 precursors: {len(precursor_data)}")
    display_limit = 50
    
    for i, (pmz, count, avg_peaks, avg_intensity) in enumerate(precursor_data):
        if i >= display_limit and len(precursor_data) > display_limit:
            omitted = len(precursor_data) - display_limit
            print(f"  ... {omitted} more precursors omitted ...")
            break
            
        print(f"  precursor_mz={pmz} | count={count} | "
              f"avg_peaks={avg_peaks:.1f} | avg_total_intensity={avg_intensity:.1f}")

def main():
    parser = argparse.ArgumentParser(
        description="Inspect mzML file: metadata, profile mode, and MS2 precursors"
    )
    parser.add_argument("mzml_files", nargs='+', help="Path to input mzML file(s)")
    parser.add_argument("--ms2", action="store_true",
                        help="Also summarize MS2 precursor metrics")
    args = parser.parse_args()

    for mzml_file in args.mzml_files:
        print(f"\n{'=' * 40}")
        print(f"FILE: {mzml_file}")
        print(f"{'=' * 40}")
        
        # Extract and display experiment metadata
        exp_metadata = get_experiment_metadata(mzml_file)
        print("EXPERIMENT METADATA:")
        print(f"  Ionization technique: {exp_metadata['ionization_technique']}")
        print(f"  Instrument: {exp_metadata['instrument']}")
        print(f"  Manufacturer: {exp_metadata['manufacturer']}")
        
        # Load MS1 spectra first
        ms1 = list(load_from_mzml(mzml_file, ms_level=1))
        
        # If polarity is unknown from file metadata, count per spectrum
        if exp_metadata['polarity'] == "Unknown":
            polarity_counts = count_polarities_in_spectra(ms1)
            print(f"  Polarity (from spectra): Positive: {polarity_counts['positive']}, "
                  f"Negative: {polarity_counts['negative']}, "
                  f"Unknown: {polarity_counts['unknown']}")
        else:
            print(f"  Ionization polarity: {exp_metadata['polarity']}")
        
        print()
        summarize_spectra(ms1, ms_level=1)

        # Optional MS2 metrics
        if args.ms2:
            ms2 = list(load_from_mzml(mzml_file, ms_level=2))
            
            # Get collision energy statistics - simplified interface
            ce_counts = count_collision_energies(ms2)
            
            # Display collision energy summary
            print("\nMS2 COLLISION ENERGY SUMMARY:")
            if sum(ce_counts.values()) > 0:
                # Sort by count (descending)
                for ce, count in sorted(ce_counts.items(), key=lambda x: -x[1]):
                    print(f"  {ce}: {count} spectra")
            else:
                print("  No collision energy information available")
            
            # Continue with regular MS2 summary
            print()
            summarize_spectra(ms2, ms_level=2)
            
            # Also report MS2 polarity if global polarity is unknown
            if exp_metadata['polarity'] == "Unknown":
                polarity_counts = count_polarities_in_spectra(ms2)
                if sum(polarity_counts.values()) > 0:
                    print(f"  MS2 Polarity: Positive: {polarity_counts['positive']}, "
                          f"Negative: {polarity_counts['negative']}, "
                          f"Unspecified: {polarity_counts['unknown']}")
            
            ms2_precursor_metrics(ms2)

if __name__ == "__main__":
    main()
