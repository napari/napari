import json
import pathlib

data = json.loads(
    (pathlib.Path(__file__).parent / 'string_list.json').read_text()
)


SKIP_FOLDERS = data['SKIP_FOLDERS']
SKIP_FILES = data['SKIP_FILES']
SKIP_WORDS_GLOBAL = data['SKIP_WORDS_GLOBAL']
SKIP_WORDS = data['SKIP_WORDS']
