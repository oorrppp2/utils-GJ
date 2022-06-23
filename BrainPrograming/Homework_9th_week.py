"""
    GUI
"""

# 5 12

# 5
# from tkinter import *
#
# total = 0
# def button_down_callback():
#     global total
#     total -= 1
#     label['text'] = total
# def button_up_callback():
#     global total
#     total += 1
#     label['text'] = total
#
# window = Tk()
#
# button_down = Button(window, text='감소', command=button_down_callback, width=20, height=5)
# label = Label(window, width=20, height=5)
# button_up = Button(window, text='증가', command=button_up_callback, width=20, height=5)
#
# button_down.grid(row=0, column=0)
# label.grid(row=0, column=1)
# button_up.grid(row=0, column=2)
#
# window.mainloop()

# 12
from tkinter import *

rect_center = [500, 300]
def button_down_callback():
    global rect_center
    global i
    rect_center[1] += 5
    w.coords(i, rect_center[0] - 50, rect_center[1] - 50,
                       rect_center[0] + 50, rect_center[1] + 50,)
def button_up_callback():
    global rect_center
    global i
    rect_center[1] -= 5
    w.coords(i, rect_center[0] - 50, rect_center[1] - 50,
                       rect_center[0] + 50, rect_center[1] + 50,)
def button_right_callback():
    global rect_center
    global i
    rect_center[0] += 5
    w.coords(i, rect_center[0] - 50, rect_center[1] - 50,
                       rect_center[0] + 50, rect_center[1] + 50,)
def button_left_callback():
    global rect_center
    global i
    rect_center[0] -= 5
    w.coords(i, rect_center[0] - 50, rect_center[1] - 50,
                       rect_center[0] + 50, rect_center[1] + 50,)

window = Tk()
w = Canvas(window, width=1000, height=600)
w.grid(row=0, columnspan=4)
i = w.create_rectangle(rect_center[0] - 50, rect_center[1] - 50,
                       rect_center[0] + 50, rect_center[1] + 50, fill="red")

button_down = Button(window, text='V(down)', command=button_down_callback)
button_up = Button(window, text='^(up)', command=button_up_callback)
button_left = Button(window, text='<-(left)', command=button_left_callback)
button_right = Button(window, text='->(right)', command=button_right_callback)
button_left.grid(row=1, column=0)
button_right.grid(row=1, column=1)
button_up.grid(row=1, column=2)
button_down.grid(row=1, column=3)
window.mainloop()

