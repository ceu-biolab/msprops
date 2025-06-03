#!/usr/bin/env python3
"""
mirror_plot.py

A script to generate a mirror plot of MS/MS fragmentation patterns from two spectral files.
Supported formats: mzML, mgf, msp (via pyteomics).

Usage:
    python mirror_plot.py file1.(mzML|mgf|msp) file2.(mzML|mgf|msp) [--precursor MZ] [--tol TOL] [--output OUT]

If one of the files contains exactly one MS/MS spectrum, its precursor m/z
will be used automatically to find the matching spectrum in the other file.
Otherwise, specify --precursor.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyteomics import mzml, mgf
import os


def load_spectra(path):
    print(f"Loading spectra from {path}")
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'mzml':
        reader = mzml.MzML(path)
        spectra = [spec for spec in reader if 'precursorList' in spec]
    elif ext == 'mgf':
        # Load MGF file with pyteomics and add more robust options
        try:
            spectra = list(mgf.read(path, use_index=False))
            print(f"Loaded {len(spectra)} spectra from {path}")
            # Try to debug the first spectrum
            if spectra and len(spectra) > 0:
                print(f"First spectrum keys: {list(spectra[0].keys())}")
        except Exception as e:
            print(f"Error loading MGF file {path}: {e}")
            raise
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return spectra


def get_precursor_mz(spectrum):
    """Extract precursor m/z from spectrum with extensive debugging"""
    try:
        # mzML
        if 'precursorList' in spectrum:
            ion = spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
            return ion['selected ion m/z']
        
        # Access params dictionary directly
        params = spectrum.get('params', {})
        
        # Check for precursor_mz in params (as seen in test.mgf)
        if 'precursor_mz' in params:
            value = params['precursor_mz']
            print(f"Found precursor_mz in params: {value}")
            # Handle string value
            if isinstance(value, str):
                return float(value)
            # Handle tuple/list value
            elif isinstance(value, (list, tuple)):
                return float(value[0])
            return float(value)
        
        # Check for pepmass in params (as seen in output.mgf)
        if 'pepmass' in params:
            value = params['pepmass']
            print(f"Found pepmass in params: {value}")
            # Handle tuple/list value (pepmass is often a tuple with intensity as second value)
            if isinstance(value, (list, tuple)):
                return float(value[0])
            return float(value)
            
        # Continue with case-insensitive checks for other cases
        # MGF direct keys (case-insensitive)
        for key in ['pepmass', 'precursor_mz', 'precursormz']:
            # Check in main spectrum dict (case-insensitive)
            for spectrum_key in spectrum:
                if spectrum_key.lower() == key:
                    value = spectrum[spectrum_key]
                    print(f"Found precursor in key '{spectrum_key}': {value}")
                    return float(value[0] if isinstance(value, (list, tuple)) else value)
        
        # MGF direct keys (exact match)
        if 'params' in spectrum:
            params = spectrum['params']
            print("Params keys:", list(params.keys()))
            for key in ['PEPMASS', 'PRECURSOR_MZ']:
                if key in params:
                    print(f"Found {key} in params: {params[key]}")
                    value = params[key]
                    return float(value[0] if isinstance(value, (list, tuple)) else value)
                
        # Try to parse from TITLE
        if 'TITLE' in spectrum:
            title = spectrum['TITLE']
            print(f"Trying to parse from TITLE: {title}")
            # Some MGF files encode precursor m/z in title like: "351.217712402343977_-1.0_scanId=4667_output"
            parts = title.split('_')
            if len(parts) > 0:
                try:
                    return float(parts[0])
                except ValueError:
                    pass
        
        # Manual parsing of spectrum entries for precursor
        for key, value in spectrum.items():
            print(f"Key: {key}, Value: {value}")
            # Last desperate attempt - just search for strings containing precursor values
            if key == 'PRECURSOR_MZ' or key == 'PEPMASS':
                print(f"Found direct precursor key: {key} = {value}")
                return float(value[0] if isinstance(value, (list, tuple)) else value)
        
        # Debug what we have - this helps diagnose the issue
        print(f"DEBUG: Available params: {list(params.keys())}")
        for key, value in params.items():
            print(f"DEBUG: {key} = {value} (type: {type(value)})")
        
        raise KeyError('Precursor m/z not found')
    except Exception as e:
        print(f"Error in get_precursor_mz: {e}")
        raise


def get_spectrum_ions(spectrum):
    mz = None
    intensity = None
    # pyteomics mzML
    if 'm/z array' in spectrum and 'intensity array' in spectrum:
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
    else:
        mz = spectrum.get('m/z array') or spectrum.get('mz') or spectrum.get('m/z_array')
        intensity = spectrum.get('intensity array') or spectrum.get('intensity') or spectrum.get('intensity_array')
    return np.array(mz), np.array(intensity)


def find_by_precursor(spectra, precursor, tol):
    """Find spectrum by precursor m/z with debugging"""
    print(f"Looking for precursor {precursor} with tolerance {tol}")
    found_precursors = []
    
    for i, spec in enumerate(spectra):
        try:
            mz0 = get_precursor_mz(spec)
            found_precursors.append(mz0)
            if abs(mz0 - precursor) <= tol:
                print(f"Found matching spectrum with precursor {mz0}")
                return spec
        except KeyError:
            print(f"Could not extract precursor m/z from spectrum {i}")
            continue
    
    print(f"Found these precursors: {found_precursors}")
    print(f"No spectrum matched precursor {precursor} within tolerance {tol}")
    return None


def plot_mirror(mz1, int1, mz2, int2, precursor, tol, out_path, file1_name, file2_name):
    # Normalize intensities
    i1 = int1 / np.max(int1) * 100 if np.max(int1) > 0 else int1
    i2 = int2 / np.max(int2) * 100 if np.max(int2) > 0 else int2

    fig, ax = plt.subplots(figsize=(10, 6), sharex=True)
    
    # Use file names as labels
    file1_label = os.path.basename(file1_name)
    file2_label = os.path.basename(file2_name)
    
    # Create stem plots without specifying linewidth
    markerline1, stemlines1, baseline1 = ax.stem(mz1, i1, linefmt='C0-', markerfmt=' ', basefmt=' ', label=file1_label)
    markerline2, stemlines2, baseline2 = ax.stem(mz2, -i2, linefmt='C1-', markerfmt=' ', basefmt=' ', label=file2_label)
    
    # Set line width for stem lines after creation
    plt.setp(stemlines1, 'linewidth', 0.8)
    plt.setp(stemlines2, 'linewidth', 0.8)
    
    ax.axvline(precursor, color='grey', linestyle='--', linewidth=0.8)

    # Add labels for peaks above 5% threshold
    for mz, intensity in zip(mz1, i1):
        if intensity > 5:  # 5% threshold
            ax.text(mz, intensity + 3, f"{mz:.1f}", ha='center', va='bottom', fontsize=8, rotation=45)
            
    for mz, intensity in zip(mz2, i2):
        if intensity > 5:  # 5% threshold
            ax.text(mz, -intensity - 3, f"{mz:.1f}", ha='center', va='top', fontsize=8, rotation=45)

    ax.set_xlabel('m/z')
    ax.set_ylabel('Relative Intensity (%)')
    ax.set_title(f'Mirror Plot — Precursor m/z: {precursor:.4f} ± {tol}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    # Ensure we save before showing (in case plt.show() blocks)
    plt.savefig(out_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate mirror plot of two MS/MS spectra')
    parser.add_argument('file1', help='First MS/MS file (mzML, mgf, msp)')
    parser.add_argument('file2', help='Second MS/MS file')
    parser.add_argument('--precursor', type=float, help='Precursor m/z to match')
    parser.add_argument('--tol', type=float, default=0.01, help='m/z tolerance')
    parser.add_argument('--output', default='mirror_plot.png', help='Output image file')
    args = parser.parse_args()

    specs1 = load_spectra(args.file1)
    specs2 = load_spectra(args.file2)
    
    print(f"Loaded {len(specs1)} spectra from {args.file1}")
    print(f"Loaded {len(specs2)} spectra from {args.file2}")

    # Determine precursor
    if args.precursor:
        precursor_mz = args.precursor
        print(f"Using user-supplied precursor: {precursor_mz}")
        spec_ref = None
    else:
        if len(specs1) == 1:
            spec_ref = specs1[0]
        elif len(specs2) == 1:
            spec_ref = specs2[0]
        else:
            parser.error('Multiple spectra in both files; please specify --precursor')
        precursor_mz = get_precursor_mz(spec_ref)

    # Find matching spectra
    if spec_ref and spec_ref in specs1:
        spec1 = spec_ref
        spec2 = find_by_precursor(specs2, precursor_mz, args.tol)
    elif spec_ref and spec_ref in specs2:
        spec2 = spec_ref
        spec1 = find_by_precursor(specs1, precursor_mz, args.tol)
    else:
        # user-specified precursor: search both ways
        spec1 = find_by_precursor(specs1, precursor_mz, args.tol)
        spec2 = find_by_precursor(specs2, precursor_mz, args.tol)

    # Try to directly access spectra if only one is available
    if not spec1 and len(specs1) == 1:
        print("Using the only spectrum in first file")
        spec1 = specs1[0]
    
    if not spec2 and len(specs2) == 1:
        print("Using the only spectrum in second file")
        spec2 = specs2[0]

    if spec1 is None or spec2 is None:
        parser.error(f'Could not find matching spectrum for precursor {precursor_mz} ± {args.tol}')

    mz1, int1 = get_spectrum_ions(spec1)
    mz2, int2 = get_spectrum_ions(spec2)
    plot_mirror(mz1, int1, mz2, int2, precursor_mz, args.tol, args.output, args.file1, args.file2)


if __name__ == '__main__':
    main()
