import argparse
import cv2
import mss
import keyboard
import numpy as np
from sys import exit
import threading

import bots








def toggle_auto_start():
    global bot
    bot.auto_start = not bot.auto_start


def toggle_auto_control():
    global bot
    bot.auto_control = not bot.auto_control


def stop_bot():
    global bot
    bot.stop()


def main():
    global bot

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        action="store",
        choices=["fisch", "dig-it"],
        help="Which mode to run the bot in.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Prints detected monitors and saves screenshots of each monitor to check if the correct monitor is selected.",
    )

    args = parser.parse_args()

    if args.debug:
        from pathlib import Path
        from PIL import Image

        image_path = Path("debug_images/")
        image_path.mkdir(exist_ok=True)

        with mss.mss() as sct:
            print("Current monitor index:", config.monitor_index)
            print("Monitors by index:")

            for i, monitor in enumerate(sct.monitors):
                if i == 0:
                    print(f"(ALL) { i }: { monitor["width"] }x{ monitor["height"] }")
                else:
                    print(f"{ i }: { monitor["width"] }x{ monitor["height"] }")

                monitor = sct.monitors[config.monitor_index]
                monitor_image = np.array(sct.grab(monitor))

                monitor_image = cv2.cvtColor(monitor_image, cv2.COLOR_BGR2RGB)
                Image.fromarray(monitor_image).save(image_path / f"monitor_{ i }.png")

        exit()

    if args.mode == "fisch":
        raise NotImplementedError()
    elif args.mode == "dig-it":
        try:
            config = bots.DigItConfig()
            config.load("configs/dig_it.yaml")
        except Exception as e:
            print(f"Failed to load config: { e }")
            exit()

        bot = bots.DigIt(config)


    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_start)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_control)

    thread = threading.Thread(target=bot.run, daemon=True)
    thread.start()


    bot.stop()
    thread.join()


if __name__ == "__main__":
    main()
