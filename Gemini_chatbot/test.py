import subprocess

espeak_path = r"C:\Program Files (x86)\eSpeak\command_line\espeak.exe"
subprocess.run([espeak_path, "-v", "en", "Hello, this is a test."])
