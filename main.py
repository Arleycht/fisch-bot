import argparse
from typing import Dict
import cv2
import keyboard
import mss
import numpy as np
import threading
import time
import tkinter as tk

import bots

modes: Dict[str, bots.Bot] = {
    "Fisch": bots.FischBot(bots.FischConfig("configs/fisch.yaml")),
    "Dig It": bots.DigItBot(bots.DigItConfig("configs/dig_it.yaml")),
}

ui_root = tk.Tk()
ui_root.config(bg="black")

default_image = tk.PhotoImage(file="base_icon.png")
failsafe_image = tk.PhotoImage(file="sleeping_icon.png")
image_label = tk.Label(ui_root, image=default_image, bg="black")

ui_root.iconphoto(True, default_image)

mode_label = tk.Label(ui_root, fg="#FFBB44", bg="black", font=(None, 16))
mode_variable = tk.StringVar(ui_root, next(iter(modes)))
mode_options = tk.OptionMenu(ui_root, mode_variable, *modes.keys())
mode_options.config(
    fg="white",
    bg="black",
    highlightthickness=0,
    activeforeground="white",
    activebackground="#222222",
    relief="raised",
    indicatoron=False,
)
mode_options["menu"].config(fg="white", bg="black")

auto_start_label = tk.Label(ui_root, text="Auto-start", fg="white", bg="black")
auto_start_status_label = tk.Label(ui_root, text="OFF", fg="white", bg="black")

auto_control_label = tk.Label(ui_root, text="Auto-control", fg="white", bg="black")
auto_control_status_label = tk.Label(ui_root, text="OFF", fg="white", bg="black")

debug_button = tk.Button(ui_root, text="Save Debug Pictures", fg="white", bg="black")

bot = modes[next(iter(modes))]


def save_debug_pictures():
    global bot

    from pathlib import Path
    from PIL import Image

    image_path = Path("debug_images/")
    image_path.mkdir(exist_ok=True)

    with mss.mss() as sct:
        print("Current monitor index:", bot.config.monitor_index)
        print("Monitors by index:")

        for i, monitor in enumerate(sct.monitors):
            if i == 0:
                print(f"(ALL) { i }: { monitor["width"] }x{ monitor["height"] }")
            else:
                print(f"{ i }: { monitor["width"] }x{ monitor["height"] }")

            monitor = sct.monitors[bot.config.monitor_index]
            monitor_image = np.array(sct.grab(monitor))

            monitor_image = cv2.cvtColor(monitor_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(monitor_image).save(image_path / f"monitor_{ i }.png")


def update():
    global bot

    mode = mode_variable.get()

    if mode not in modes:
        raise ValueError("Invalid mode")
    elif modes[mode] != bot:
        bot = modes[mode]
        bot.run()
    elif not bot.is_alive():
        bot.run()

    mode_label.config(text=f"{ mode } Mode")

    if bot.auto_start:
        auto_start_status_label.config(text="---[ON]", fg="green")
    else:
        auto_start_status_label.config(text="[OFF]---", fg="red")

    if bot.auto_control:
        auto_control_status_label.config(text="---[ON]", fg="green")
    else:
        auto_control_status_label.config(text="[OFF]---", fg="red")


def toggle_auto_start():
    global bot
    bot.toggle_auto_start()
    update()


def toggle_auto_control():
    global bot
    bot.toggle_auto_control()
    update()


def main():
    global bot

    image_label.pack()
    mode_label.pack()
    mode_options.pack()
    auto_start_label.pack()
    auto_start_status_label.pack()
    auto_control_label.pack()
    auto_control_status_label.pack()
    debug_button.pack(pady=20)

    ui_root.bind("<Control-w>", lambda _: ui_root.quit())
    debug_button.config(command=save_debug_pictures)
    mode_variable.trace_add("write", lambda *_: update())

    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_start)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_control)

    def check_thread_status():
        global bot

        while True:
            time.sleep(1)

            if not bot.is_alive():
                update()

            if bot.failsafe_active:
                image_label.config(image=failsafe_image)
            else:
                image_label.config(image=default_image)

    update()

    status_thread = threading.Thread(target=check_thread_status, daemon=True)
    status_thread.start()

    ui_root.mainloop()


if __name__ == "__main__":
    main()
