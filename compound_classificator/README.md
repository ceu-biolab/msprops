# Compound classifier

`classify_compounds.py` reads a text file with one compound per line and writes
compound classifications as JSON or CSV.

Accepted input identifiers:

- InChIKey
- SMILES
- InChI

Blank lines and lines starting with `#` are ignored.

## Setup

Install the HTTP dependency:

```bash
pip install -r compound_classificator/requirements.txt
```

SMILES and InChI inputs also require RDKit. InChIKey-only input does not:

```bash
pip install rdkit-pypi
```

## Services

You must choose at least one service with `--services`; there is no default.

Available services:

- `classyfire`: submits compounds to ClassyFire and returns ChemOnt taxonomy.
  If ClassyFire does not return a classification, the script automatically
  tries FiehnLab as a ChemOnt fallback for those missing compounds.
- `fiehnlab`: queries the FiehnLab ClassyFire browser directly.
- `chebi`: queries ChEBI through OLS and maps the result through a local
  ChEBI ontology file.
- `pubchem`: queries PubChem classification data for compounds still missing
  after the selected primary services.

ChEBI requires a local ontology file, which is too large to keep in this
repository:

1. Download `chebi.obo` from <https://www.ebi.ac.uk/chebi/downloads>
2. Place it at `compound_classificator/chebi.obo`

## Usage

Write one compound identifier per line:

```text
CC(=O)O
InChI=1S/H2O/h1H2
QTBSBXVTEAMEQO-UHFFFAOYSA-N
```

Recommended ChemOnt workflow, using ClassyFire with automatic FiehnLab fallback:

```bash
python compound_classificator/classify_compounds.py compounds.txt --services classyfire -o classifications.json
```

Use FiehnLab directly:

```bash
python compound_classificator/classify_compounds.py compounds.txt --services fiehnlab -o classifications.json
```

Write CSV output:

```bash
python compound_classificator/classify_compounds.py compounds.txt --services classyfire --output-format csv -o classifications.csv
```

Add PubChem as an extra fallback:

```bash
python compound_classificator/classify_compounds.py compounds.txt --services classyfire --enable-pubchem -o classifications.json
```

Use ChEBI after placing `chebi.obo` in `compound_classificator/`:

```bash
python compound_classificator/classify_compounds.py compounds.txt --services chebi -o classifications.json
```
