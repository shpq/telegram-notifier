from .image_utils import (
    get_plots,
    obj2imagebytes,
)
from omegaconf import OmegaConf
from dotenv import load_dotenv, find_dotenv
from .utils import get_error_message
import telegram
import logging
import os

load_dotenv(find_dotenv())


class Telegram:
    """
    Telegram bot class. Using to send messages and photos to
    your chatbot
    """

    def __init__(self, token: str, chat_id: str, project=None):
        if token is None or chat_id is None:
            self.bot = None
        else:
            self.bot = telegram.Bot(token=token)

        self.chat_id = chat_id
        self.project = project

    def val2list(self, values, length: int = 1):
        if not isinstance(values, list):
            values = [values]
        return values * length

    def set_creds(self, token, chat_id):
        """
        Set bot token and chat_id
        """
        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id

    def send_plots(self, values):
        """
        Send plots of training values depending on epoch num
        """
        values = self.val2list(values)
        if isinstance(values, list):
            [self.send_plot(v) for v in values if v]

    def send_plot(self, values):
        """
        Send plot if it can be plotted
        """
        image_bytes = get_plots(values)
        if self.bot is not None and image_bytes is not None:
            self.bot.send_photo(photo=image_bytes, chat_id=self.chat_id)

    def send_photo(self, image, norm, size=None, save_path=None):
        """
        Resize and send image with custom normalization, save in save_path
        folder with simple enumerate naming
        """
        image_bytes = obj2imagebytes(image, norm, size, save_path)
        try:
            self.bot.send_photo(photo=image_bytes, chat_id=self.chat_id)
        except Exception as e:
            message = get_error_message(e)
            logging.debug(f"cannot send image, got exception:\n{message}")

    def send_images(self, images, norm=None, size=None, save_path=None):
        """
        Send np.array or PIL.Image images / list of images to bot
        with renormalization
        """
        images = self.val2list(images)
        norm = self.val2list(norm, length=len(images))
        size = self.val2list(size, length=len(images))
        for image, n, s in zip(images, norm, size):
            self.send_photo(image, norm=n, size=s, save_path=save_path)

    def send_message(self, message=None):
        """
        Send text message to bot
        """
        logging.info(message)
        if self.project is not None:
            message = f"{self.project}:\n" + message
        if self.bot is not None:
            try:
                self.bot.send_message(text=message, chat_id=self.chat_id, timeout=10)
            except:
                pass

    def send_project_config(self, cfg):
        """
        Send project config to bot
        """
        self.send_message(message=OmegaConf.to_yaml(cfg))

    def set_project(self, project):
        """
        Set project name
        """
        self.project = project

bot = Telegram(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"))
