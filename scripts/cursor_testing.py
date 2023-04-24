from psychopy import visual, core, event
import helper_functions as hf

cursor_size = 0.1
target_size = 0.1
home_size = 0.15
home_range_size = home_size * 5

win = visual.Window(fullscr=True, monitor='testMonitor',
                    units='pix', color='black', waitBlanking=False, screen=1, size=[1920, 1080])

int_cursor = visual.Circle(
    win, radius=hf.cm_to_pixel(cursor_size), fillColor='Black')  # integrated pos

print(hf.cm_to_pixel(cursor_size))
print(int_cursor.radius)