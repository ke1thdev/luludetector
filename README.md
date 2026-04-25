<div align="center">
  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"></path>
    <path d="M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"></path>
    <path d="M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0"></path>
    <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"></path>
  </svg>
  
  <h1>Hand Motion Detector</h1>
  <p>A computer vision python application that tracks up-and-down motion patterns to trigger auditory feedback.</p>
</div>

<hr>

## Overview

This application uses your device's webcam to track hand movements in real-time. By utilizing MediaPipe, the script maps the coordinates of your hand and calculates vertical velocity data over a rolling time window. When it detects a sustained and rhythmic oscillation (up and down) passing dynamic thresholds, it triggers an MP3 file playback.

## Features

- Real-time hand landmark tracking and overlay visualization.
- Resolves dynamic hand scale to ensure motion detection math works regardless of the user's distance from the webcam.
- Built-in velocity deadzones and smoothings to ignore simple camera jitter.
- Application-level cooldown mechanism preventing overlapping triggers.

## Stack Requirements

- **Python** (Core logic and math routines)
- **OpenCV** (Webcam hardware interface, matrix operations, and drawing the HUD)
- **MediaPipe** (Real-time machine learning architecture for multi-hand coordinate mapping)
- **Pygame** (Audio mixer initialization and low-latency playback)

## Installation

You must have Python installed. Clone the repository to your local machine, then install the strict dependencies via pip.

```sh
pip install -r requirements.txt
```

*(Note: The dependencies include `opencv-python`, `mediapipe`, and `pygame`.)*

## Usage

1. Ensure the target audio file is present in same root directory. By default, the application looks for `wa-na-nag-lulu-na.mp3`.
2. Run the script from your terminal:

```sh
python luludetector.py
```

3. A new interface window will open, streaming your webcam feed.
4. Position your primary hand clearly in the frame until the tracking markers (green nodes, orange connecting lines) appear.
5. Move your hand up and down continuously. The top left corner will show `Motion: DETECTED` once the threshold validates the action, followed by the audio playback.
6. Press the **`q`** key on your keyboard while the window is focused to release all hooks and gracefully shutdown.

## Configuration

If the detector feels too sensitive, or isn't firing quickly enough, you can find the `Tunable Settings` block at the top of the `luludetector.py` file. Important variables include:

* `WINDOW_SECONDS`: How far back (in seconds) the application looks to detect the rhythmic pattern.
* `MIN_SIGN_CHANGES`: Minimum times the hand must change direction in the chronological buffer.
* `MIN_SPEED_NORM`: Set a higher threshold to enforce faster movements before classifying matches.
* `COOLDOWN_SECONDS`: The lockout duration after playback where no actions will trigger sound.

## Security Notice

The script requires direct webcam access to function. Camera processing happens strictly locally in machine memory; no streams or recording bytes are dispatched over any network.
