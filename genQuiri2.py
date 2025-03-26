import time
import subprocess

while True:
    # Run the genQeuri.py script for 1 hour
    process = subprocess.Popen(['python', 'genQeuri.py'])
    
    time.sleep(100) 
    process.terminate()  # Stop the script

    time.sleep(30)  # Wait for 30 seconds before restarting