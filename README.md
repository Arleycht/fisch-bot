## About

A proof of concept automated fishing script for the Roblox game Fisch using computer vision.

Also known as a fisch macro, fisch bot, fisch script.

## Setup and Usage

It is recommended to use a Python virtual environment to run this script.

- Open a command prompt terminal in the repository directory and run `python -m venv .venv` to create a virtual environment in the `.venv` directory.
- Activate the virtual environment: `.venv\Scripts\activate.bat`
- Install the requirements: `pip install -r requirements.txt`.
- Run the script: `python main.py`.
  - Terminate it by closing the command prompt window or put the window into focus and press `Ctrl+C` to break out of the script.

Currently, the numbers are configured for a single 1920x1080 monitor with the Roblox window maximized (not fullscreen).

There is a second configuration available by switching out the commented alternative reel rects where the window is docked to the right. You can dock it to the right by either dragging the window to the right of the screen or pressing `Windows Key + Right Arrow`.

Hotkeys are currently configured as so:

- `Ctrl+Shift+C` to toggle auto-casting
- `Ctrl+Shift+F` to toggle auto-shaking
- `Ctrl+Shift+R` to toggle auto-reeling

## Methodology

Auto-casting is trivial because it does not affect catches, so we set it to click and release on a fixed time interval.

Auto-shake is implemented through OpenCV's template matching function. The source and target images are given a $3 \times 3$ Gaussian blur and run through the Sobel filter as according to the [OpenCV example](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html) before template matching. The reasoning for this is because the button background is transparent ($\alpha \approx 0.6$), therefore it is more reliable to match the edges. In practice, the threshold for reliable matches was determined to be around $0.45$. Auto-shake is slightly sensitive to network latency as it delays the appearance of the shake button, which is difficult to discern from a fish being caught or the fishing rod to be retracted. This in combination with auto-casting makes the script retract the line when the button does not appear in time because it attempts to auto-cast. Future iterations could detect whether the player has retracted the line or not by either observing the player or fishing rod, possibly utilizing a pre-trained image model since the dataset should be trivial to obtain as positive/negative images of the player with the line out or not. It is likely more practical to simply construct a better methodology such as waiting for a shake button to appear before some timeout expires before being sure that the player does not have the line out.

Auto-reeling uses a PID controller with gains set to $(P, D, I) = (1.5, 0.45, 0.05)$ and integral error clipped to $[-0.5, 0.5]$. The detection of the the reeling state is done by using a hard-coded crop of the window screenshot. The bar positions are obtained by taking the bounding box of the non-zero pixels in the image after taking values within experimentally determined ranges of a grayscale image. The fish usually lies within $[70, 80]$, and the reel being within $[v_{max} - 10, 255]$ where $v_{max}$ is the brightest pixel value in the image. This method has the drawback of being somewhat fragile to changes in value and sometimes fails depending on the background behind the reeling UI, as it is slightly transparent. Future iterations could be reworked to account for a slightly larger cropped image that finds the fish position through the contour or the fish icon. The reel position appears to be robust in practice.

## Possible Improvements

- Not hard coding screen rects.
- Line casting is currently set to a fixed timer rather than searching for the cast meter because it changes size depending on the camera distance. This can probably be solved using some sort of scale invariant feature search to get the bar position, then measuring the pixels vertically.
- Some sort of visual feedback for what toggles are currently on.
