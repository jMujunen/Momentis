# Momentis

## Overview

Momentis is a Python-based project that aims to analyze first-person shooter video game clips for specific keywords (names in kill-feed) and patterns (audio ques) using OpenCV and other libraries. The main goal of this project is to take several hours of footage from individual clips and trim them down leaving only the relevant parts.

## Usage

As always, using a virtual environment is recommended:

```sh
python -m venv venv
# Linux
source venv/bin/activate
# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
python moviepy_writer.py /path/to/your/videos
```

This will process the videos in the specified directory and generate output in a subdirectory named "opencv-output". The output includes the resulting trimmed videos along with .json debugging files for each processed video.
