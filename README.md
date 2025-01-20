## About

Automated digging script for the Roblox game Dig It using computer vision.

## Setup and Usage

It is recommended to use a Python virtual environment to run this script.

- Open a command prompt terminal in the repository directory and run `python -m venv .venv` to create a virtual environment in the `.venv` directory.
- Activate the virtual environment: `.venv\Scripts\activate.bat`
- Install the requirements: `pip install -r requirements.txt`.
- Run the script: `python main.py`.
  - Terminate it by closing the command prompt window or put the window into focus and press `Ctrl+C` to break out of the script.
- You will have to activate the virtual environment before running the script if you are in a fresh command prompt terminal, so it is recommended to use the terminal in an editor (like Code) that will automatically activate the Python virtual environment.

There are a few additional considerations for setting up the configuration if you are already familiar with the Fisch variant.

`edge_rect` should be the area in the black border that is equal width to the inside of the bar, but positioned on the outline. The cursor that the player controls should be the only thing moving left and right across the black outline in this area.
`sample_coord` should be a position that is in the colored outline and it shouldn't be in the way of the sideways moving cursor. This coordinate will be used to sample the color of the target bar.
