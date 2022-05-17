import tkinter as tk


WIDTH, HEIGHT = 600, 800


def draw(event):
    inwindow.create_oval(
        event.x - 16,
        event.y - 16,
        event.x + 16,
        event.y + 16,
        fill='black',
    )


root = tk.Tk()

inwindow = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
# for x in range(WIDTH // 100):
#     for y in range(HEIGHT // 100):
#         inwindow.create_line(x, y, x, y + HEIGHT, fill='grey')

inwindow.bind('<B1-Motion>', draw)

inwindow.pack()

root.mainloop()
