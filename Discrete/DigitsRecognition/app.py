import tkinter as tk
from PIL import ImageGrab
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ann import ANN

class App(object):
    def __init__(self, image_size=(16, 16), model_ANN=ANN):
        self.image_size = image_size
        self.ann = model_ANN(self.image_size)

    def load_model_ANN(self, path=None):
        self.ann.load(path)

    def get_UI(self, title='title', brush_size=10, brush_color='black'):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('800x600')
        self.root.resizable(width=False, height=False)

        self.brush_size = brush_size
        self.brush_color = brush_color
        self.text_predict_label = 'Цифра неопознана'

        self.frame_paint = tk.LabelFrame(self.root, font=('Arial', 14), text='Полотно для рисования / Изображение подаваемое на обработку')
        self.frame_paint.pack(expand=1, fill=tk.BOTH, padx=0, pady=10)

        self.canv_paint = tk.Canvas(self.frame_paint, bg='white', highlightthickness=2, highlightbackground='black', width=360, height=360)
        self.canv_paint.bind('<B1-Motion>', self._draw) 
        self.canv_paint.pack(side=tk.LEFT, padx=10, pady=10)

        self.fig = Figure(figsize=(5, 5))
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.image = self.ax.imshow([[0]])
        self.canv_image = FigureCanvasTkAgg(self.fig, master=self.frame_paint)
        self.canv_image.get_tk_widget().config(bg='white', highlightthickness=2, highlightbackground='black', width=360, height=360)
        self.canv_image.get_tk_widget().pack(side=tk.RIGHT, padx=10, pady=10)
        self.canv_image.draw()

        self.frame_button = tk.Frame(self.root)
        self.frame_button.pack(side=tk.BOTTOM, padx=10, pady=10)

        self.start_button = tk.Button(self.frame_button, font=('Arial', 16), text='Предсказать цифру', bg='#CCCCFF', padx=10, pady=5, width=18, command=self._model_prediction)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.predict_label = tk.Label(self.frame_button, font=('Arial', 14), text=self.text_predict_label, bg='#A8E4A0', padx=10, pady=5, width=20, borderwidth=2, relief='solid')
        self.predict_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.clear_button = tk.Button(self.frame_button, font=('Arial', 16), text='Отчистить все', bg='#CCCCFF', padx=10, pady=5, width=18, command=self._clear_all)
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.root.mainloop()

    def _draw(self, event):
        self.canv_paint.create_oval(
            event.x - self.brush_size,
            event.y - self.brush_size,
            event.x + self.brush_size,
            event.y + self.brush_size,
            fill=self.brush_color, outline=self.brush_color,
        )

    def _model_prediction(self):
        new_image = ImageGrab.grab(bbox=(
            self.canv_paint.winfo_rootx() + 2,
            self.canv_paint.winfo_rooty() + 2,
            self.canv_paint.winfo_rootx() + self.canv_paint.winfo_width() - 2,
            self.canv_paint.winfo_rooty() + self.canv_paint.winfo_height() - 2,
        ))

        self.image.set_data(new_image.resize(self.image_size))
        self.canv_image.draw()

        self.predict_label.config(text=f'{self.ann.predict(new_image)}')

    def _clear_all(self):
        self.canv_paint.delete('all')
        self.predict_label.config(text=self.text_predict_label)
        self.image.set_data([[0]])
        self.canv_image.draw()