"""

This script gets the transcript files from all the subdirectories returned by coursera-dl.

Coursera files are originall structured in subdirectories like the tree below. It would be easier to work with the data if we stored them in one or two files.
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
import shutil
from pathlib import Path

def main():
    
    # modify to to where the coursera-dl directories are located
    data_dir = Path(r'/Users/pzuradzki/Dropbox/Documents/Hub/2021/uiuc/fall2021/cs410-text-information-systems/team-project/cs410_team_shared_data_store')

    assert 'text-retrieval' in os.listdir(data_dir), 'missing course files for text-retrieval'
    assert 'text-mining' in os.listdir(data_dir), 'missing course files for text-mining'

    # we need to copy the *.en.srt and *.en.txt English transcript files to one location
    srt_rgx = re.compile(r'(.*en.srt)')
    txt_rgx = re.compile(r'(.*en.txt)')

    # make coursera_transcripts directory if it does not already exist
    (data_dir / 'coursera_transcripts').mkdir(parents=True, exist_ok=True)
    (data_dir / 'coursera_transcripts/txt').mkdir(parents=True, exist_ok=True)
    (data_dir / 'coursera_transcripts/srt').mkdir(parents=True, exist_ok=True)


    # call copy func 4 times; 2 MOOCs (text retrieval, text mining) * 2 file types (txt, srt)
    copy_txt_files_to_target(source_dir=data_dir / 'text-retrieval',
                             rgx_pattern=r'(.*en.txt)',
                             target_dir=data_dir / 'coursera_transcripts/txt')

    copy_txt_files_to_target(source_dir=data_dir / 'text-mining',
                             rgx_pattern=r'(.*en.txt)',
                             target_dir=data_dir / 'coursera_transcripts/txt')

    copy_txt_files_to_target(source_dir=data_dir / 'text-retrieval',
                             rgx_pattern=r'(.*en.srt)',
                             target_dir=data_dir / 'coursera_transcripts/srt')

    copy_txt_files_to_target(source_dir=data_dir / 'text-mining',
                             rgx_pattern=r'(.*en.srt)',
                             target_dir=data_dir / 'coursera_transcripts/srt')

def copy_txt_files_to_target(source_dir, rgx_pattern, target_dir):
    """Walks a source directory recursively and copies .txt or .srt transcript files to a target directory.

    Parameters
    ----------
    source_dir: source directory that contains files nested in subdirectories
    rgx_pattern: regular expression pattern to determine which files to copy. Use r'(.*en.srt)' or r'(.*en.txt)' 
    target_dir: destination directory for copied files
    """

    rgx = re.compile(rgx_pattern) # compiled regex    
    src_filepaths = []

    for root, dirs, files in os.walk(data_dir / 'text-retrieval'):
        for file in files:
            if rgx.search(file) is not None:
                src_filepaths.append(Path(root) / file)

    src_filepaths = sorted(src_filepaths)
    
    for filepath in srt_filepaths:
        shutil.copy2(filepath, target_dir / filepath.name)
    
if __name__ == "__main__":
    main()