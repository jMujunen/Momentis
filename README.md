# Momentis

## Overview

Momentis is a Python-based project that aims to analyze first-person shooter video game
clips for specific keywords (names in kill-feed) and patterns (audio ques) using OpenCV and other libraries.
The main goal of this project is to take several hours of footage from individual clips
and trim them down leaving only the relevant parts.

To determine the efficacy of this project, the following show simple metrics comparing
the total duration and size of the original footage with the trimmed versions.

| Game  | Original Duration | Original Size | Result Duration     | Result Size | Diff               |
| ----- | ----------------- | ------------- | ------------------- | ----------- | ------------------ |
| CS:GO | 8 hours           | 26GB          | **2 hours 45 mins** | **8GB**     | **~ 3x Reduction** |
| CS2   | TODO              | TODO          | TODO                | TODO        | TODO               |
| PUBG  | TODO              | TODO          | TODO                | TODO        | TODO               |
| Apex  | TODO              | TODO          | TODO                | TODO        | TODO               |

> Note: This will re-encoding each video and due to limitations in `moviepy` api, gpu transcoding is not supported.
> This results in a long processing time for large video sets.
> For reference, 8 hours of CS:GO footage took my i7-12700k roughly 4 hours. I suggest running it overnight

## Usage

As always, using a virtual environment is recommended:

1. Clone this repository
2. Create a virtual environment
3. Activate your virtual environment
4. Install dependencies
5. Populate keywords.txt with desired words to initialize a subclip (one per line)
6. Run Momentis

```bash
# Clone this repository
git clone https://github.com/jMujunen/momentis.git
cd momentis
python -m venv venv

# Linux
source venv/bin/activate
# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
python momentis.py /path/to/your/videos
```

This will process the videos in the specified directory and generate output in a subdirectory named "opencv-output". The output includes the resulting trimmed videos along with .json debugging files for each processed video.

### Populating keywords.txt

Keywords are used to identify when to start and stop recording a subclip. Several subclips create a single video. You should populate keywords.txt with words that you expect to see in the kill-feed of your game, such as your username.
Generally, the _OCR_ (Optical character recognition) has a hard time finding an exact match in certain scenes so its recommended use small 'chunks' instead a of a whole word.

## Showcase

The following shows a glimpse of how it works

Original Duration: **90s**
Trimmed Duration: **41s**

![](./assets/example_intput.gif)

Below is the resulting video

![](./assets/_example.gif)
