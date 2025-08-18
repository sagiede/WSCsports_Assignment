# WSC Project

A Jupyter notebook project for data analysis and NLP tasks.

## Setup

1. Create virtual environment:
```bash
python -m venv wsc_venv
source wsc_venv/bin/activate  # On Windows: wsc_venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy language model:
```bash
python -m spacy download en_core_web_lg
```

4. Register Jupyter kernel:
```bash
python -m ipykernel install --user --name=wsc_venv --display-name="WSC Environment"
```

## Environment

- Python 3.9.6
- NLP libraries: nltk, transformers, torch
- Data science: pandas, numpy, seaborn, matplotlib
- Jupyter notebook support

## Usage

Open `notebook.ipynb` and select the "WSC Environment" kernel.
