from __future__ import annotations

from threading import Thread
from IPython.display import display
from PIL import Image

from ..app.parameters import Parameters


def async_display(image: Image.Image, parameters: Parameters) -> Thread:
    def print_display(image: Image.Image, parameters: Parameters) -> None:
        print(parameters)
        display(image)

    return Thread(target=print_display, args=(image, parameters))
