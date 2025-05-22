import sys
from collections import Counter
from lxml import etree

def count_collision_energies(mzml_path):
    ns = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    counts = Counter()
    context = etree.iterparse(mzml_path, events=('end',), tag='{http://psi.hupo.org/ms/mzml}spectrum')
    for event, elem in context:
        ms_level = None
        for cv in elem.findall('.//mzml:cvParam', ns):
            if cv.get('name') == 'ms level':
                ms_level = cv.get('value')
                break
        if ms_level == '2':
            # Find collision energy in precursor > activation > cvParam
            for ce_cv in elem.findall('.//mzml:precursorList/mzml:precursor/mzml:activation/mzml:cvParam', ns):
                if ce_cv.get('name') == 'collision energy':
                    ce = ce_cv.get('value')
                    counts[ce] += 1
        elem.clear()
    return counts

def main():
    if len(sys.argv) < 2:
        print('Usage: python count_collision_energies.py <file1.mzML> [file2.mzML ...]')
        sys.exit(1)
    for mzml_path in sys.argv[1:]:
        print(f'File: {mzml_path}')
        try:
            counts = count_collision_energies(mzml_path)
            if counts:
                for ce, count in sorted(counts.items(), key=lambda x: float(x[0])):
                    print(f'  Collision energy {ce} eV: {count} spectra')
            else:
                print('  No MS2 collision energies found.')
        except Exception as e:
            print(f'  Error processing file: {e}')

if __name__ == '__main__':
    main()
