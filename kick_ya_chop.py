import keyboard
import mss
import cv2
import numpy as np
from time import time, sleep
import pyautogui
from rich.console import Console
from colorama import init

pyautogui.PAUSE = 0

keyboard.wait('s')
left = True
x_coords = {True: 1288, False: 1638}
y = 725
sct = mss.mss()
dimensions = {
    True: {'left': 1325, 'top': 512, 'width': 250, 'height': 250},
    False: {'left': 1525, 'top': 512, 'width': 250, 'height': 250}
}

wood_images = {side: cv2.imread(f'image/{side}.png') for side in ['left', 'right']}
wood_shapes = {key: img.shape[1::-1] for key, img in wood_images.items()}

console = Console()
init(autoreset=True)

fps_time = time()
while True:
    side = 'left' if left else 'right'
    scr = np.array(sct.grab(dimensions[left]))[:, :, :3]
    scr = cv2.cvtColor(scr, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    wood = wood_images[side]
    w, h = wood_shapes[side]

    result = cv2.matchTemplate(scr, wood, cv2.TM_CCOEFF_NORMED)
    _, maxval, _, max_loc = cv2.minMaxLoc(result)
    console.log(f"[bold green]Max Val:[/bold green] {maxval} [bold green]Max Loc:[/bold green] {max_loc}")

    if maxval > .70:
        left = not left
        cv2.rectangle(scr, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)

    cv2.imshow('Screen Shot', scr)
    cv2.waitKey(1)
    pyautogui.click(x=x_coords[left], y=y)
    sleep(.10)
    if keyboard.is_pressed('q'):
        break

    console.log(f"[bold blue]FPS:[/bold blue] {1 / (time() - fps_time):.2f}")
    fps_time = time()
