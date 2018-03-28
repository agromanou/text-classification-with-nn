from os import getcwd, pardir, makedirs
from os.path import join, abspath, exists

PARENT_DIRECTORY = abspath(join(getcwd(), pardir))
DATA_DIR = join(PARENT_DIRECTORY, 'data')
MODELS_DIR = join(PARENT_DIRECTORY, 'models')

# if the folders don't exist, create them.
if not exists(DATA_DIR):
    makedirs(DATA_DIR)

if not exists(MODELS_DIR):
    makedirs(MODELS_DIR)