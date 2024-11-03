## About

A proof of concept automated fishing script for the Roblox game Fisch using computer vision.

Also known as a fisch macro, fisch bot, fisch script.

## Possible Improvements

- Scale invariance in the shake button search.
- Not hard coding screen rects.
- Line casting is currently set to a fixed timer rather than searching for the cast meter because it changes size depending on the camera distance. This can probably be solved using some sort of scale invariant feature search to get the bar position, then measuring the pixels vertically.
- Reeling minigame can be solved using a slightly more sophisticated algorithm to nearly guarantee all catches.
- Some sort of visual feedback for what toggles are currently on.
