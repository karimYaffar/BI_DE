import keyboard
def check_for_pause(function):
    """
    Function to pause the execution when Ctrl+P is pressed.
    """
    if keyboard.is_pressed('ctrl') and keyboard.is_pressed('p'):
        while True:
            if keyboard.is_pressed('ctrl') and keyboard.is_pressed('r'):
                break
    if keyboard.is_pressed('ctrl') and keyboard.is_pressed('b'):
        exit(0)
    if keyboard.is_pressed('ctrl') and keyboard.is_pressed('e'):
        function()
        while True:
            if keyboard.is_pressed('ctrl') and keyboard.is_pressed('z'):
                break