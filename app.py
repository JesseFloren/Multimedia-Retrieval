from subprocess import call
import threading

t = threading.Thread(target=lambda _: call(["python", "./src/main.py"]), args=(1,))
t.start()
call(["streamlit", "run", "./ui.py"])