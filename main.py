import argparse
import cv2
import keyboard
import mss
import numpy as np
import threading
import time
import tkinter as tk

import bots

ui_root = tk.Tk()

ui_root.iconbitmap("main.ico")
ui_root.configure(bg="black")

default_image = tk.PhotoImage(file="base_icon.png")
failsafe_image = tk.PhotoImage(file="sleeping_icon.png")
image_label = tk.Label(ui_root, image=default_image, bg="black")

mode_label = tk.Label(ui_root, fg="#FFBB44", bg="black", font=(None, 16))

auto_start_label = tk.Label(ui_root, text="Auto-start", fg="white", bg="black")
auto_start_status_label = tk.Label(ui_root, text="OFF", fg="white", bg="black")

auto_control_label = tk.Label(ui_root, text="Auto-control", fg="white", bg="black")
auto_control_status_label = tk.Label(ui_root, text="OFF", fg="white", bg="black")

debug_button = tk.Button(ui_root, text="Save Debug Pictures", fg="white", bg="black")

current_config: bots.BotConfig
current_bot: bots.Bot


def update_labels():
    global bot

    if bot.auto_start:
        auto_start_status_label.configure(text="---[ON]", fg="green")
    else:
        auto_start_status_label.configure(text="[OFF]---", fg="red")

    if bot.auto_control:
        auto_control_status_label.configure(text="---[ON]", fg="green")
    else:
        auto_control_status_label.configure(text="[OFF]---", fg="red")


def toggle_auto_start():
    global bot
    bot.toggle_auto_start()
    update_labels()


def toggle_auto_control():
    global bot
    bot.toggle_auto_control()
    update_labels()


def stop_bot():
    global bot
    ui_root.quit()
    bot.stop()


def save_debug_pictures():
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


def main():
    image_label.pack()
    mode_label.pack()
    auto_start_label.pack()
    auto_start_status_label.pack()
    auto_control_label.pack()
    auto_control_status_label.pack()
    debug_button.pack(pady=20)

    debug_button.configure(command=save_debug_pictures)

    global bot
    global config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        action="store",
        choices=["fisch", "dig-it"],
        help="Which mode to run the bot in.",
    )

    args = parser.parse_args()

    if args.mode == "fisch":
        config = bots.FischConfig()
        config.load("configs/fisch.yaml")

        bot = bots.Fisch(config)
        mode_label.configure(text="Fisch Mode")
    elif args.mode == "dig-it":
        config = bots.DigItConfig()
        config.load("configs/dig_it.yaml")

        bot = bots.DigIt(config)
        mode_label.configure(text="Dig It Mode")
    else:
        raise ValueError("Unknown mode")

    update_labels()

    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_start)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_control)

    ui_root.bind("<Control-w>", lambda _: stop_bot())

    thread = threading.Thread(target=bot.run, daemon=True)
    thread.start()

    def check_thread_status():
        while thread.is_alive():
            if bot.failsafe_active:
                image_label.configure(image=failsafe_image)

            time.sleep(1)
        ui_root.quit()

    status_thread = threading.Thread(target=check_thread_status, daemon=True)
    status_thread.start()

    ui_root.mainloop()


if __name__ == "__main__":
    main()
