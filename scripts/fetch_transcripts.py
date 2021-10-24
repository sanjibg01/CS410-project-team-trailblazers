"""

This script gets the transcript files from all the subdirectories returned by coursera-dl.

Coursera files are originall structured in subdirectories like the tree below. It would be easier to work with the data if we stored them in one directory or one merged JSON.
.
├── 01_orientation
│   ├── 01_about-the-course
│   └── 02_orientation-activities
...
├── 07_week-6
│   ├── 01_week-6-information
│   ├── 02_week-6-lessons
│   ├── 03_week-6-activities
│   └── 04_honors-track-programming-assignment
├── files.txt
└── text-retrieval-syllabus-parsed.json
"""

import os
import re
import json
from pathlib import Path

def main():
    
    # modify to to where the coursera-dl directories are located
    data_dir = Path(r'/Users/pzuradzki/Dropbox/Documents/Hub/2021/uiuc/fall2021/cs410-text-information-systems/team-project/cs410_team_shared_data_store')

    assert 'text-retrieval' in os.listdir(data_dir), 'missing course files for text-retrieval'
    assert 'text-mining' in os.listdir(data_dir), 'missing course files for text-mining'

    # make coursera_transcripts directory if it does not already exist
    (data_dir / 'coursera_transcripts').mkdir(parents=True, exist_ok=True)

    # we need to copy the *.en.srt and *.en.txt English transcript files to one location
    # call converter func 4 times; 2 MOOCs (text retrieval, text mining) * 2 file types (txt, srt)
    filetype_rgx_lookup = {'txt': r'(.*en.txt)', 'srt': r'(.*en.srt)'}    
    for course in ['text-retrieval', 'text-mining']:
        for filetype in filetype_rgx_lookup:
            convert_transcripts_to_json(source_dir=data_dir / course,
                                        rgx_pattern=filetype_rgx_lookup[filetype],
                                        target_filepath=data_dir / f'coursera_transcripts/transcripts_{course}_{filetype}.json')

def convert_transcripts_to_json(source_dir, rgx_pattern, target_filepath):
    """Walks a source directory recursively and copies .txt or .srt transcript files to a target json / dict-like collection.

    Parameters
    ----------
    source_dir: source directory that contains files nested in subdirectories
    rgx_pattern: regular expression pattern to determine which files to copy. Use r'(.*en.srt)' or r'(.*en.txt)' 
    target_filepath: destination filepath for consolidate json file
    """

    rgx = re.compile(rgx_pattern) # compiled regex    
    src_filepaths = []

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if rgx.search(file) is not None:
                src_filepaths.append(Path(root) / file)

    src_filepaths = sorted(src_filepaths)
    
    transcripts = {}

    for file in src_filepaths:
        with open(file, 'r') as f:
            transcripts[file.name] = f.read()

    with open(target_filepath, 'w') as fp:
        json.dump(obj=transcripts, fp=fp, indent=1)
        
if __name__ == "__main__":
    main()