import argparse
import json
from lxml import etree
import pymzml

def parse_xml_metadata(file_path):
    ns = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    tree = etree.parse(file_path)
    # Software list
    software = []
    for sw in tree.xpath('//mzml:software', namespaces=ns):
        sid = sw.get('id') or ''
        ver = sw.get('version') or ''
        software.append({'id': sid, 'version': ver})
    # Instrument configurations
    instruments = []
    for inst in tree.xpath('//mzml:instrumentConfiguration', namespaces=ns):
        iid = inst.get('id') or ''
        instruments.append({'id': iid})
    # Run info
    run_elem = tree.find('.//{http://psi.hupo.org/ms/mzml}run')
    run_id = run_elem.get('id') if run_elem is not None else ''
    # Experimental parameters
    exp = {}
    # run start timestamp
    if run_elem is not None:
        st = run_elem.get('startTimeStamp')
        if st:
            exp['start_time'] = st
    # instrument configuration parameters
    instr_params = {}
    for inst in tree.xpath('//mzml:instrumentConfiguration', namespaces=ns):
        iid = inst.get('id', '')
        params = []
        for cv in inst.xpath('.//mzml:cvParam', namespaces=ns):
            params.append({
                'accession': cv.get('accession',''),
                'name': cv.get('name',''),
                'value': cv.get('value','')
            })
        if params:
            instr_params[iid] = params
    if instr_params:
        exp['instrument_configurations'] = instr_params
    # scanSettings parameters
    scan_params = []
    for cv in tree.xpath('//mzml:scanSettings//mzml:cvParam', namespaces=ns):
        scan_params.append({
            'accession': cv.get('accession',''),
            'name': cv.get('name',''),
            'value': cv.get('value','')
        })
    if scan_params:
        exp['scan_settings'] = scan_params
    # source configuration parameters
    src_params = []
    for cv in tree.xpath('//mzml:instrumentConfiguration//mzml:source//mzml:cvParam', namespaces=ns):
        src_params.append({
            'accession': cv.get('accession',''),
            'name': cv.get('name',''),
            'value': cv.get('value','')
        })
    if src_params:
        exp['source_configurations'] = src_params
    # Ionization modes
    ion = []
    for cv in tree.xpath('//mzml:instrumentConfiguration//source//mzml:cvParam', namespaces=ns):
        name = cv.get('name','')
        if 'ionization' in name.lower():
            ion.append({'name': name, 'value': cv.get('value','')})
    if ion:
        exp['ionization_modes'] = ion
    # Polarity settings
    pol = []
    for cv in tree.xpath('//mzml:scanSettings//mzml:cvParam', namespaces=ns):
        name = cv.get('name','')
        if 'polarity' in name.lower():
            pol.append({'name': name, 'value': cv.get('value','')})
    if pol:
        exp['polarities'] = pol
    return {'software': software,
            'instruments': instruments,
            'run_id': run_id,
            'experimental': exp}

def count_spectra(file_path):
    reader = pymzml.run.Reader(file_path)
    # counts per MS level and unique precursor m/z tracking
    level_counts = {}
    unique_precursors = {}
    for spec in reader:
        level = getattr(spec, 'ms_level', None)
        if level is None:
            continue
        # increment spectrum count
        level_counts[level] = level_counts.get(level, 0) + 1
        # collect precursor m/z if available
        precs = []
        if hasattr(spec, 'selected_precursors'):
            precs = spec.selected_precursors or []
        else:
            # fallback: check precursorList in native XML
            precs = getattr(spec, 'precursor_list', [])
        for p in precs:
            mz = p.get('mz') if isinstance(p, dict) else getattr(p, 'mz', None)
            if mz is None:
                continue
            unique_precursors.setdefault(level, set()).add(round(float(mz), 4))
    # prepare output dict
    result = {}
    for lvl, cnt in level_counts.items():
        result[f'ms{lvl}_count'] = cnt
    for lvl, mzs in unique_precursors.items():
        result[f'ms{lvl}_unique_precursors'] = len(mzs)
    return result

def main():
    parser = argparse.ArgumentParser(description='Describe an mzML file: metadata and spectra counts')
    parser.add_argument('mzml_file', help='Path to the mzML file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()

    meta = parse_xml_metadata(args.mzml_file)
    counts = count_spectra(args.mzml_file)
    result = {**meta, **counts}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Run ID: {result['run_id']}")
        print("Software:")
        for s in result['software']:
            print(f"  - {s['id']} (version {s['version']})")
        print("Instruments:")
        for i in result['instruments']:
            print(f"  - ID: {i['id']}")
        # print all MS-level counts and unique precursor counts
        for key in sorted(k for k in result if k.startswith('ms')):
            # format display name
            name = key.upper().replace('_', ' ')
            print(f"{name}: {result[key]}")

if __name__ == '__main__':
    main()