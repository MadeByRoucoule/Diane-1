from customtkinter import *
import tkinter as tk
import os
import glob

class Interface(CTk):
    def __init__(self, network, training_data, input_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.title("Diane・1")
        self.geometry("1020x520")
        self.resizable(False, False)

        # On stocke ici le réseau et les données mais on n'y accède pas directement pour l'entraînement
        self.network = network
        self.training_data = training_data
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Liste des poids disponibles
        weights_dir = "weights"
        self.weights_list = [os.path.basename(f) for f in glob.glob(os.path.join(weights_dir, "*.json"))]

        # Liste des fichiers de test
        tests_dir = "data/testing"
        self.tests_list = [os.path.basename(f) for f in glob.glob(os.path.join(tests_dir, "*.json"))]

        self.widgets()

    def widgets(self):
        self.left_frame = CTkFrame(self, width=500, height=500, fg_color="transparent")
        self.left_frame.pack(padx=(10,0), pady=10, side='left')

        self.digit_canvas = CTkCanvas(self.left_frame, width=496, height=496, background="gray14", highlightthickness=0)
        self.digit_canvas.place(relx=0.5, rely=0.5, anchor='center')
        self.draw_rounded_rect(self.digit_canvas, 0, 0, 496, 496, radius=35, fill="#1d1d1d")
        
        values = [
            [0.0, 0.0, 0.3125, 0.8125, 0.5625, 0.0625, 0.0, 0.0],
            [0.0, 0.0, 0.8125, 0.9375, 0.625, 0.9375, 0.3125, 0.0],
            [0.0, 0.1875, 0.9375, 0.125, 0.0, 0.6875, 0.5, 0.0],
            [0.0, 0.25, 0.75, 0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.3125, 0.5, 0.0, 0.0, 0.5625, 0.5, 0.0],
            [0.0, 0.25, 0.6875, 0.0, 0.0625, 0.75, 0.4375, 0.0],
            [0.0, 0.125, 0.875, 0.3125, 0.625, 0.75, 0.0, 0.0],
            [0.0, 0.0, 0.375, 0.8125, 0.625, 0.0, 0.0, 0.0]
        ]
        
        self.fill_grid(self.digit_canvas, values, 496, 496, 8, 8, "#1d1d1d", "#dddddd")
        self.draw_grid(self.digit_canvas, 496, 496, 8, 8, "#dddddd")

        self.right_frame = CTkFrame(self, width=500, height=500, fg_color="transparent")
        self.right_frame.pack(padx=10, pady=10, side='right')
        self.right_frame.pack_propagate(False)

        self.choice_var = tk.StringVar(value="training")

        self.choice_frame = CTkFrame(self.right_frame, width=480, height=40, corner_radius=15)
        self.choice_frame.pack(pady=5)
        self.choice_frame.pack_propagate(False)
        
        self.training_choice_radio_btn = CTkRadioButton(self.choice_frame, text='Training', variable=self.choice_var, value="training", command=self.update_frame)
        self.training_choice_radio_btn.pack(side='left', padx=(100,0), fill='x')
        
        self.already_trained_choice_radio_btn = CTkRadioButton(self.choice_frame, text='Already Trained', variable=self.choice_var, value="already_trained", command=self.update_frame)
        self.already_trained_choice_radio_btn.pack(side='right', padx=(0,100), fill='x')

        self.training_frame = CTkFrame(self.right_frame, width=480, height=170, corner_radius=20)
        self.training_frame.pack_propagate(False)

        self.epochs_frame = CTkFrame(self.training_frame, width=470, height=40, corner_radius=15, fg_color='gray22')
        self.epochs_frame.pack_propagate(False)
        self.epochs_frame.pack(pady=5)
        self.epochs_label = CTkLabel(self.epochs_frame, text="Epochs :")
        self.epochs_label.pack(padx=(10,5), side='left')
        self.epochs_entry = CTkEntry(self.epochs_frame)
        self.epochs_entry.pack(padx=5, side='right', expand=True, fill='x')
        self.epochs_entry.insert(0, "10 000")
        self.epochs_entry.bind("<KeyRelease>", self.format_number)
        self.training_pb = CTkProgressBar(self.training_frame, width=470, height=10, corner_radius=15, fg_color='gray22', progress_color='#226622')
        self.training_pb.pack(pady=5)
        self.training_infos_frame = CTkFrame(self.training_frame, width=470, height=40)
        self.training_infos_frame.pack(pady=5)
        self.training_infos_frame.pack_propagate(False)
        self.training_pourcent_label = CTkLabel(self.training_infos_frame, text=f'{45}%')
        self.training_pourcent_label.pack(padx=10, side='left')
        self.training_remain_time_label = CTkLabel(self.training_infos_frame, text=f'Time remain : {23}s')
        self.training_remain_time_label.pack(padx=10, side='right')
        self.training_btn = CTkButton(self.training_frame, text="Train Network", width=470, height=40, corner_radius=15, fg_color='gray25', hover_color='gray30')
        self.training_btn.pack(pady=5)

        self.already_trained_frame = CTkFrame(self.right_frame, width=480, height=100, corner_radius=20)
        self.already_trained_frame.pack_propagate(False)

        self.weights_file_menu = CTkOptionMenu(self.already_trained_frame, values=self.weights_list, width=470, height=40, corner_radius=15, fg_color='gray22', button_color='gray25', button_hover_color='gray30')
        self.weights_file_menu.pack(pady=5)
        self.set_weights_file_btn = CTkButton(self.already_trained_frame, text="Set weights", width=470, height=40, corner_radius=15, fg_color='gray25', hover_color='gray30')
        self.set_weights_file_btn.pack(pady=5)

        self.separator = CTkProgressBar(self.right_frame, width=480, height=2, fg_color='gray25', progress_color='gray25')
        self.separator.pack(pady=10)

        self.test_frame = CTkFrame(self.right_frame, width=480, corner_radius=20)
        self.test_frame.pack(pady=5, fill='y', expand=True)
        self.test_frame.pack_propagate(False)
        
        self.test_settings_frame = CTkFrame(self.test_frame, corner_radius=15, fg_color='gray20')
        self.test_settings_frame.pack(padx=5, pady=5, side='left', expand=True, fill='both')
        self.test_settings_frame.pack_propagate(False)

        self.test_file_menu = CTkOptionMenu(self.test_settings_frame, values=self.tests_list, height=40, corner_radius=10, fg_color='gray25', button_color='gray27', button_hover_color='gray30')
        self.test_file_menu.pack(padx=5, pady=5, fill='x')
        self.random_frame = CTkFrame(self.test_settings_frame, height=40, corner_radius=10, fg_color='gray25')
        self.random_frame.pack(padx=5, pady=5, fill='x')
        self.random_check_btn = CTkCheckBox(self.random_frame, text="Random index", height=30, fg_color='#226622', hover_color='#226622', command=self.toggle_entry)
        self.random_check_btn.pack(padx=5, pady=5, fill='x')
        self.random_check_btn.select()
        self.test_intex_entry = CTkEntry(self.test_settings_frame, placeholder_text="Index", height=40, corner_radius=10)
        self.test_intex_entry.pack(padx=5, pady=5, fill='x')
        self.random_test_btn = CTkButton(self.test_settings_frame, text="Test network", height=40, corner_radius=10, fg_color='gray25', hover_color='gray30')
        self.random_test_btn.pack(padx=5, pady=5, fill='x', side='bottom')
        separator = CTkProgressBar(self.test_settings_frame, height=2, fg_color='gray30', progress_color='gray30')
        separator.pack(padx=10, pady=5, fill='x', side='bottom')

        self.test_result_frame = CTkFrame(self.test_frame, corner_radius=15, fg_color='gray20')
        self.test_result_frame.pack(padx=5, pady=5, side='right', expand=True, fill='both')
        self.test_result_frame.pack_propagate(False)

        self.test_output_frame = CTkFrame(self.test_result_frame, height=50, fg_color='gray25')
        self.test_output_frame.pack(padx=5, pady=5, side='top', fill='x')
        self.test_output_frame.pack_propagate(False)

        self.test_pourcent_frame = CTkScrollableFrame(self.test_result_frame, fg_color='gray25')
        self.test_pourcent_frame.pack(padx=5, pady=5, side='top', expand=True, fill='both')


        self.toggle_entry()
        self.update_frame()

    def update_frame(self):
        choix = self.choice_var.get()
        if choix == "training":
            self.training_frame.pack(pady=5, before=self.separator)
            self.already_trained_frame.pack_forget()
        elif choix == "already_trained":
            self.already_trained_frame.pack(pady=5, before=self.separator)
            self.training_frame.pack_forget()

    def draw_rounded_rect(self, canvas, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                  x1+radius, y1, x2-radius, y1,
                  x2-radius, y1, x2, y1,
                  x2, y1+radius, x2, y1+radius,
                  x2, y2-radius, x2, y2-radius,
                  x2, y2, x2-radius, y2,
                  x2-radius, y2, x1+radius, y2,
                  x1+radius, y2, x1, y2,
                  x1, y2-radius, x1, y2-radius,
                  x1, y1+radius, x1, y1+radius,
                  x1, y1]
        return canvas.create_polygon(points, **kwargs, smooth=True)

    def draw_grid(self, canvas, width, height, rows, cols, color):
        row_height = height // rows
        col_width = width // cols

        for i in range(1, rows):
            canvas.create_line(0, i * row_height, width, i * row_height, fill=color)

        for i in range(1, cols):
            canvas.create_line(i * col_width, 0, i * col_width, height, fill=color)

    def fill_grid(self, canvas, values, width, height, rows, cols, bg_color, fg_color):
        row_height = height // rows
        col_width = width // cols

        bg_color_rgb = self.hex_to_rgb(bg_color)
        fg_color_rgb = self.hex_to_rgb(fg_color)

        for i in range(rows):
            for j in range(cols):
                if values[i][j] == 0:
                    continue
                x1 = j * col_width
                y1 = i * row_height
                x2 = x1 + col_width
                y2 = y1 + row_height
                color = self.interpolate_color(bg_color_rgb, fg_color_rgb, values[i][j])
                canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

    def interpolate_color(self, color1, color2, factor):
        r = int(color1[0] + (color2[0] - color1[0]) * factor)
        g = int(color1[1] + (color2[1] - color1[1]) * factor)
        b = int(color1[2] + (color2[2] - color1[2]) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def format_number(self, event):
        entry = event.widget
        value = entry.get().replace(" ", "")
        if value.isdigit():
            formatted_value = "{:,}".format(int(value)).replace(",", " ")
            entry.delete(0, tk.END)
            entry.insert(0, formatted_value)

    def toggle_entry(self):
        if self.random_check_btn.get() == 1:
            self.test_intex_entry.configure(state='disabled')
        else:
            self.test_intex_entry.configure(state='normal')