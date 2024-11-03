## About

A proof of concept automated fishing script for the Roblox game Fisch using computer vision.

Also known as a fisch macro, fisch bot, fisch script.

## Setup and Usage

It is recommended to use a Python virtual environment to run this script.

You can do this by opening a command prompt terminal in the repository directory and running `python -m venv .venv` to create a virtual environment in the `.venv` directory.

To activate the virtual environment, you can then run `.venv\Scripts\activate.bat`

Install the requirements using `pip install -r requirements.txt`

Currently, the numbers are configured for a single 1920x1080 monitor with the Roblox window is docked to the right. You can dock it to the right by either dragging the window to the right of the screen or pressing `Windows Key + Right Arrow`.

Hotkeys are currently configured as so:

- `Ctrl+Shift+C` to toggle auto-casting
- `Ctrl+Shift+F` to toggle auto-shaking
- `Ctrl+Shift+R` to toggle auto-reeling

## Possible Improvements

- Not hard coding screen rects.
- Line casting is currently set to a fixed timer rather than searching for the cast meter because it changes size depending on the camera distance. This can probably be solved using some sort of scale invariant feature search to get the bar position, then measuring the pixels vertically.
- Some sort of visual feedback for what toggles are currently on.
