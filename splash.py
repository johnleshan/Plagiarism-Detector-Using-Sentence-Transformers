import tkinter as tk
import subprocess

# Create splash screen
splash = tk.Tk()
splash.overrideredirect(True)  # Remove window borders
splash.attributes("-topmost", True)  # Keep splash on top

# Set window size and position (adjust as needed)
width, height = 300, 300
screen_width = splash.winfo_screenwidth()
screen_height = splash.winfo_screenheight()
x, y = (screen_width - width) // 2, (screen_height - height) // 2
splash.geometry(f"{width}x{height}+{x}+{y}")

# Load splash image
try:
    img = tk.PhotoImage(file="splash.png")  
    label = tk.Label(splash, image=img, bg="black")  
    label.pack()
except tk.TclError as e:
    print("Error loading splash image:", e)
    splash.destroy()

# Start app.py and close splash screen
def start_app():
    subprocess.Popen(["python", "app.py"])  # Start app.py
    splash.destroy()  # Close splash immediately

splash.after(500, start_app)  # Start app after 0.5 seconds

splash.mainloop()
