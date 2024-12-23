import requests
import threading
from dataclasses import dataclass
import random
from textwrap import fill as word_wrap
import signal

OPENAI_API_HOST = "http://localhost:1234"
MODEL_NAME = "Llama-3.1-8B-Lexi-Uncensored-V2-GGUF"


class Bot:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.temperature = random.randint(7, 12) / 10.0
        self.history = []

        self.print_history = False

        print(f"SYSTEM: Bot '{self.name}' created with desc: {self.desc} and creativity: {self.temperature}")

    def send_message(self, message, role, other_name, other_type):
        message = {"role": role, "content": message}
        system_guidance = {
            "role": "system",
            "content": f"You are {self.name}, a {self.desc}. You are talking to {other_name}, a {other_type}. Don't repeat what {other_name} says. Don't reply with similar answers, or looping conversations. Try to move the conversation forward in different directions. Answer as concisely as possible.",
        }

        self.history.append(message)
        self.history.append(system_guidance)

        if self.print_history:
            print("---")
            print(self.name, "history:")
            for entry in self.history:
                print(entry)
            print("---")

        response = requests.post(
            f"{OPENAI_API_HOST}/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": self.history[-20:], "temperature": self.temperature},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Unexpected response from server. Status code: {response.status_code}:", response.text)

        response = response.json()
        response = response["choices"][0]["message"]

        self.history.append(response)

        return response["content"]


SYSTEM_WARNING1 = {
    "role": "system",
    "content": "Move to conversation to a different direction or topic, it is becoming stagnant",
}
SYSTEM_WARNING2 = {
    "role": "system",
    "content": "Answer in less than 20 words",
}


class Conversation:
    def __init__(self, bot1, bot2):
        self.bot1 = bot1
        self.bot2 = bot2
        self.scene_desc = None
        self.initial_bot1_message = None
        self.history = []

    def run(self):
        print("")

        for bot in (self.bot1, self.bot2):
            initial_message = f"Your name is {bot.name}. You are a {bot.desc}. The scene description is: {self.scene_desc}. Answer in 50 words or less."
            print(f"SYSTEM to {bot.name}: {initial_message}")

            initial_message = {"role": "system", "content": initial_message}

            bot.history.append(initial_message)

        message = self.initial_bot1_message
        bot = self.bot1

        while True:
            self.history.append({"name": bot.name, "content": message})

            response = word_wrap(message, width=80, subsequent_indent="\t")
            print(f"{bot.name} ({bot.temperature: 0.1f}): {response}")
            print("---")

            other_bot = bot
            bot = self.bot1 if bot == self.bot2 else self.bot2

            message = bot.send_message(message, "user", other_bot.name, other_bot.desc)

            if len(self.history) % 6 == 0:
                other_bot.history.append(SYSTEM_WARNING1)
                # print(">> conversation fork warning")
            if len(self.history) % 5 == 0:
                other_bot.history.append(SYSTEM_WARNING2)
                # print(">> conversation length warning")


@dataclass
class Scene:
    name1: str = ""
    desc1: str = ""

    name2: str = ""
    desc2: str = ""

    scene_desc: str = ""
    init_bot1_msg: str = ""


scene = Scene()


def main():
    scene.name1 = input("Enter the name of Bot 1: ")
    scene.desc1 = input("Enter the description of Bot 1: ")

    scene.name2 = input("Enter the name of Bot 2: ")
    scene.desc2 = input(f"Enter the description of Bot 2: ")

    scene.scene_desc = input("Enter the scene description: ")
    scene.init_bot1_msg = input(f"Narrator, what does {scene.name1} say? ")

    bot1 = Bot(scene.name1, scene.desc1)
    bot2 = Bot(scene.name2, scene.desc2)

    conversation = Conversation(bot1, bot2)
    conversation.scene_desc = scene.scene_desc
    conversation.initial_bot1_message = scene.init_bot1_msg

    thread = threading.Thread(target=conversation.run)
    thread.daemon = True
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
