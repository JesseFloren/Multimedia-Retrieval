from subprocess import call
import threading
import src.main as main

t = threading.Thread(target=lambda _: call(["streamlit", "run", "./ui.py"]), args=(1,))
t.start()
main.main("Run")