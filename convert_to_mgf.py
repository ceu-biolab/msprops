from pyopenms import MSExperiment, MzMLFile, MascotGenericFile
import argparse

def convert_mzml_to_mgf(input_file: str, output_file: str):
    exp = MSExperiment()
    mzml_file = MzMLFile()
    mzml_file.load(input_file, exp)

    mgf = MascotGenericFile()
    mgf.store(output_file, exp)

def main():
    parser = argparse.ArgumentParser(description='Convert mzML file to MGF format.')
    parser.add_argument('input_file', help='Path to the input mzML file')
    parser.add_argument('output_file', help='Path to the output MGF file')
    args = parser.parse_args()

    convert_mzml_to_mgf(args.input_file, args.output_file)
    print(f"Converted {args.input_file} to {args.output_file}")

if __name__ == "__main__":
    main()
