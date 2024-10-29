## Overview

Momentis is a Python-based project that aims to analyze first-person shooter video game
clips for specific keywords (names in kill-feed) and patterns (audio ques: TODO) using OpenCV and other libraries.
The main goal of this project is to take several hours of footage from individual clips
and trim them down leaving only the relevant parts.

To determine the efficacy of this project, the following show simple metrics comparing
the total duration and size of the original footage with the trimmed versions.

| Game  | Original Duration | Original Size | Result Duration     | Result Size | Diff           |
| ----- | ----------------- | ------------- | ------------------- | ----------- | -------------- |
| CS:GO | 8 hours           | 26GB          | **2 hours 45 mins** | **8GB**     | ~ 3x Reduction |
| PUBG  | 3 hours 45 mins   | 46GB          | **30 mins**         | **7GB**     | ~ 7x Reduction |
| CS2   | TODO              | TODO          | TODO                | TODO        | TODO           |
| Apex  | TODO              | TODO          | TODO                | TODO        | TODO           |

> Note: This will re-encoding each video and due to limitations in `moviepy` api, gpu transcoding is not supported.
> This results in a long processing time for large video sets.
> For reference, 8 hours of CS:GO footage took my i7-12700k roughly 4 hours. I suggest running it overnight. Expect high cpu usage.

### Getting Started

These instructions will get you up and running on your local machine.

Steps:

1. Clone this repository
2. Create a virtual environment
3. Activate your virtual environment
4. Install dependencies
5. Populate keywords.txt with desired words to initialize a subclip (one per line)
6. Run Momentis

#### Prerequisites

- Python >= 3.12
- python (executable in your PATH)
- git (executable in your PATH)

To ensure python has been installed correctly run the following command:

```python
$ python --version
Python 3.12.6
```

#### Setup

As always, using a virtual environment is recommended:

##### Recommended Steps

```bash
# Step 1: Clone this repository
git clone https://github.com/jMujunen/momentis.git
cd momentis
# Step 2: Create a virtual environment (optional but recommended)
python -m venv venv

# Step 3: Activate your virtual environment

# Linux
source venv/bin/activate
# Windows
\venv\Scripts\activate

# Step 4: Install dependencies
pip install -r requirements.txt
```

##### Alternative Steps

1. Download the zip file by clicking on "Code" -> "Download ZIP"
2. Extract the zip file into any location on your computer
3. Open a shell eg. command prompt, terminal, bash etc.

- Navigate to where you extracted the files using `cd /path/to/momentis-master`
- Othewise, open a shell in your file explorer by right clicking the Momentis-master
  and choosing "Open in terminal" or "Open PowerShell window here"

4. Activate the virtual environment (optional but recommended)

```bash
# Linux
source venv/bin/activate
# Windows
\venv\Scripts\activate
```

5. Install dependencies

```bash
pip install -r requirements.txt
```

#### Populating keywords.txt

Keywords are used to identify when to start and stop recording a subclip. Several subclips create a single video. You should populate keywords.txt with words that you expect to see in the kill-feed of your game, such as your username.
Generally, the _OCR_ (Optical character recognition) has a hard time finding an exact match in certain scenes so its recommended use small 'chunks' instead a of a whole word.

For the clip shown at the bottom, my username is 'MrHoffman'. The following shows how my username gets interpreted by the OCR:

```text
mrhoffman_
mrhoffman_
you finally killed
you knocked out
hrhoffman.
mihoffman_
mrhafiman
```

#### Running Momentis

```bash
python momentis/momentis.py /path/to/your/videos
```

This will process the videos in the specified directory and generate output in a subdirectory named "opencv-output"

## Showcase

The following shows a glimpse of how it works

Original Duration: **90s**
Trimmed Duration: **41s**

![](./assets/example_intput.gif)

Below is the resulting video

![](./assets/_example.gif)
