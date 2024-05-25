import os

INPUT_DATA_PATH = os.path.join(os.getcwd(), 'zenodo_png', 'input')
LABEL_DATA_PATH = os.path.join(os.getcwd(), 'zenodo_png', 'lung_label')
INPUT_DIR = os.listdir(INPUT_DATA_PATH)
PATH_LIST = []
for i in range(len(INPUT_DIR)):
    dir = INPUT_DIR[i]
    filepaths = []
    for file in os.listdir(os.path.join(INPUT_DATA_PATH, dir)):
        filepaths.append((os.path.join(INPUT_DATA_PATH, dir, file), os.path.join(LABEL_DATA_PATH, dir, file)))
    PATH_LIST.append(filepaths)

MODEL_PATH = os.path.join(os.getcwd(), 'models')

