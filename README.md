## About

Automated fishing script for the Roblox game Fisch using computer vision.

Also known as a fisch macro, fisch bot, fisch script.

## Setup and Usage

It is recommended to use a Python virtual environment to run this script.

- Open a command prompt terminal in the repository directory and run `python -m venv .venv` to create a virtual environment in the `.venv` directory.
- Activate the virtual environment: `.venv\Scripts\activate.bat`
- Install the requirements: `pip install -r requirements.txt`.
- Run the script: `python main.py`.
  - Terminate it by closing the command prompt window or put the window into focus and press `Ctrl+C` to break out of the script.
- You will have to activate the virtual environment before running the script if you are in a fresh command prompt terminal, so it is recommended to use the terminal in an editor (like Code) that will automatically activate the Python virtual environment.

There is an AFK fail-safe for auto-cast if no shake nor reel minigame is detected after 60 seconds. This will allow Roblox's default AFK timer to tick until completion.

To untrip the fail-safe, you can manually cast your fishing rod to pop up the shake or reel minigame and it will reset the fail-safe trigger.

If no shake nor reel minigame is detected after 10 minutes, the script will automatically break out of the loop and terminate. These failsafes will also print their time of occurrence in the case that it might line up with an expected server shutdown or otherwise.

Hotkeys are currently configured as so:

- `Ctrl+Shift+C` to toggle auto-casting
- `Ctrl+Shift+F` to toggle auto-shaking
- `Ctrl+Shift+R` to toggle auto-reeling

## Configuring

Hard coded values only need to be configured if you are not on a single 1920x1080 reoslution monitor. Otherwise, you will have grab two screenshots: One with the shake button to measure its diameter, and another to measure the rectangle that the reelbar takes up along with the associated text prompt above the bar and a box that surrounds it. Note that rectangles as described here are their top left corner position $(x, y)$ and their width and height $(w, h)$.

The file `reel_prompt.png` should be a cropped transparent image of just the text "Click and Hold Anywhere!" at the correct size as it appears over the reel bar.

The variable `reel_prompt_rect` should be the $(x, y, w, h)$ of the rectangle that generally surrounds the reel prompt as it appears on screen.

The variable `reel_rect` should be the $(x, y, w, h)$ of the rectangle that the reel control bar can move within.

The variable `monitor_index` will not necessarily match with what Windows indexes the monitors as, so `main.py` includes a `debug_mode` variable which, if set to `True` and when you run the script, will save an image into the same directory called `full_preview.png`. This image should be of a single monitor or screen, so you can change `monitor_index`, save your change, and run the script again until it is the correct screen. Once you're done, remember to set `debug_mode` back to `False` to run the script normally.

A video demonstration of configuring these numbers will be provided in the future, but for now I hope that these text instructions can suffice.

## Methodology

Because casting quality doesn't affect anything, auto-cast clicks on a fixed timer in a known safe position on the screen.

Auto-shake reconstructs an ideal shake button and searches for multiple sizes in a screenshot of the Roblox window. The reason for this is to support the Steady Rod, which features a large shake button. Because the coordinates are grabbed from a screenshot of the window, auto-shake will work regardless of configured positions as long as the sizes are configured correctly. Because of the slight transparency, the match algorithm is supplemented by using Sobel filtered images instead of color. While the reel bar offsets can be done the same way, it isn't because that's just how the proof of concept was designed so it hasn't changed because it works.

Auto-reel is currently set up using fixed positions that must be manually determined. Generally, the images are filtered using a combination of range and morphology filters to extract relevant rectangles to obtain a normalized position for the reel controller. The PID controller coefficients are tuned to the gains of $(P, D, I) = (1, 0.5, 0)$, and if the target is closer to an edge by half the width of the control bar, it uses $(P, D, I) = (1, 0, 0)$.

There is an additional correction ratio dependent on the forces on the control bar since there is a distinct force for both directions. An estimator obtains these accelerations and the ratio between them is used to scaled the error such that high rightward forces will be less sensitive to errors, and the reverse.

## Possible Improvements

- Not hard coding screen rects.
- Line casting is currently set to a fixed timer rather than searching for the cast meter because it changes size depending on the camera distance. This can probably be solved using some sort of scale invariant feature search to get the bar position, then measuring the pixels vertically.
- Some sort of visual feedback for what toggles are currently on.
