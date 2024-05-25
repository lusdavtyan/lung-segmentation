## Getting started

Clone the project

```bash
  git clone https://github.com/lusdavtyan/lung-segmentation.git
```
Download *COVID-19-CT-Seg_20cases.zip*, *Lung_Mask.zip* from https://zenodo.org/record/3757476#.YWG1tRDMLyJ

Go to the project directory

```bash
  cd lung_segmentation
```

Install dependencies

```bash
  pip3 install -r requirements.txt
```

To run the FastAPI application

```bash
  cd app
  python fastapi_main.py
```