from customtkinter import *
import tkinter as tk
import threading
import random
import glob
import os

class Interface(CTk):
    def __init__(self, network, training_data, input_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.title("Diane・1")
        self.geometry("1020x520")
        self.resizable(False, False)

        self.network = network
        self.training_data = training_data
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.update_vars()

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

        self.tabs = CTkTabview(self.right_frame, width=500, height=500, segmented_button_selected_color='#226622', segmented_button_selected_hover_color='#115511', segmented_button_unselected_hover_color='#335533')
        self.tabs.pack(fill='both', expand=True)

        self.training_tab = self.tabs.add("Training")
        self.testing_tab = self.tabs.add("Testing")

        self.training_frame = CTkFrame(self.training_tab)
        self.training_frame.pack_propagate(False)
        self.training_frame.pack(fill='both', expand=True)
        
        self.training_title = CTkLabel(self.training_frame, text="Training", font=("", 16, 'bold'), anchor='w')
        self.training_title.pack(padx=15, pady=5, fill='x')
        
        self.epochs_frame = CTkFrame(self.training_frame, width=470, height=40, corner_radius=10, fg_color='gray22')
        self.epochs_frame.pack_propagate(False)
        self.epochs_frame.pack(padx=5, pady=5)

        self.epochs_label = CTkLabel(self.epochs_frame, text="Epochs :")
        self.epochs_label.pack(padx=(10,5), side='left')
        self.epochs_entry = CTkEntry(self.epochs_frame)
        self.epochs_entry.pack(padx=5, side='right', expand=True, fill='x')
        self.epochs_entry.insert(0, "10 000")
        self.epochs_entry.bind("<KeyRelease>", self.format_number)

        self.save_weights_frame = CTkFrame(self.training_frame, width=470, height=40, corner_radius=15, fg_color='gray22')
        self.save_weights_frame.pack_propagate(False)
        self.save_weights_frame.pack(padx=5, pady=5)
        self.save_weights_check_btn = CTkCheckBox(self.save_weights_frame, text="Save weights", height=30, fg_color='#226622', hover_color='#335533')
        self.save_weights_check_btn.pack(padx=10, pady=5, fill='x')
        self.save_weights_check_btn.select()

        self.training_bottom_frame = CTkFrame(self.training_frame, width=470, corner_radius=15)
        self.training_bottom_frame.pack(padx=5, pady=5, side='bottom')

        self.training_infos_frame = CTkFrame(self.training_bottom_frame, width=470, height=40, corner_radius=15, fg_color='gray25')
        self.training_infos_frame.pack(padx=5, pady=5)
        self.training_infos_frame.pack_propagate(False)
        self.training_pourcent_label = CTkLabel(self.training_infos_frame, text=f'{100}%')
        self.training_pourcent_label.pack(padx=10, side='left')
        self.training_remain_time_label = CTkLabel(self.training_infos_frame, text=f'Time remain : {0}s')
        self.training_remain_time_label.pack(padx=10, side='right')

        self.training_pb = CTkProgressBar(self.training_bottom_frame, width=470, height=15, corner_radius=15, fg_color='gray25', progress_color='#226622')
        self.training_pb.pack(padx=5, pady=5)
        self.training_pb.set(1)

        self.training_btn = CTkButton(self.training_bottom_frame, text="Train Network", width=470, height=40, corner_radius=15, fg_color='gray25', hover_color='gray30', command=self.train_network)
        self.training_btn.pack(padx=5, pady=5)

        self.testing_frame = CTkFrame(self.testing_tab)
        self.testing_frame.pack_propagate(False)
        self.testing_frame.pack(fill='both', expand=True)

        self.testing_title = CTkLabel(self.testing_frame, text="Testing", font=("", 16, 'bold'), anchor='w')
        self.testing_title.pack(padx=15, pady=5, fill='x')

        self.testing_left_frame = CTkFrame(self.testing_frame, corner_radius=15)
        self.testing_left_frame.pack_propagate(False)
        self.testing_left_frame.pack(padx=5, pady=5, side='left', fill='both', expand=True)

        self.weights_file_menu = CTkOptionMenu(self.testing_left_frame, values=self.weights_list, width=470, height=40, corner_radius=10, fg_color='gray27', button_color='gray30', button_hover_color='gray35')
        self.weights_file_menu.pack(padx=5, pady=5)

        self.test_file_menu = CTkOptionMenu(self.testing_left_frame, values=self.tests_list, height=40, corner_radius=10, fg_color='gray27', button_color='gray30', button_hover_color='gray35')
        self.test_file_menu.pack(padx=5, pady=5, fill='x')

        self.random_frame = CTkFrame(self.testing_left_frame, height=40, corner_radius=10, fg_color='gray25')
        self.random_frame.pack(padx=5, pady=5, fill='x')
        self.random_check_btn = CTkCheckBox(self.random_frame, text="Random index", height=30, fg_color='#226622', hover_color='#335533', command=self.toggle_entry)
        self.random_check_btn.pack(padx=5, pady=5, fill='x')
        self.random_check_btn.select()
        self.test_index_entry = CTkEntry(self.testing_left_frame, placeholder_text="Index", height=40, corner_radius=10)
        self.test_index_entry.pack(padx=5, pady=5, fill='x')

        self.network_test_btn = CTkButton(self.testing_left_frame, text="Test network", height=40, corner_radius=10, fg_color='gray25', hover_color='gray30', command=self.test_network)
        self.network_test_btn.pack(padx=5, pady=5, fill='x', side='bottom')

        self.testing_right_frame = CTkFrame(self.testing_frame, corner_radius=15)
        self.testing_right_frame.pack_propagate(False)
        self.testing_right_frame.pack(padx=5, pady=5, side='right', fill='both', expand=True)

        self.testing_outputs_frame = CTkFrame(self.testing_right_frame, height=90, corner_radius=10, fg_color='gray25')
        self.testing_outputs_frame.pack(padx=5, pady=5, fill='x')
        self.testing_outputs_frame.grid_propagate(False)
        self.testing_outputs_frame.columnconfigure(0, weight=1)
        self.testing_outputs_frame.columnconfigure(2, weight=1)

        self.testing_output_computed_frame = CTkFrame(self.testing_outputs_frame, corner_radius=10, fg_color='transparent')
        self.testing_output_computed_frame.grid(row=0, column=0, sticky='nsew', padx=(5,0), pady=5)

        self.testing_output_computed_label = CTkLabel(self.testing_output_computed_frame, text="Computed output")
        self.testing_output_computed_label.pack(fill='x')

        self.testing_output_computed_value = CTkLabel(self.testing_output_computed_frame, text="5", font=("", 40, 'bold'))
        self.testing_output_computed_value.pack(fill='x')
        
        self.separator = CTkProgressBar(self.testing_outputs_frame, width=2, height=80, orientation='vertical', fg_color='gray30', bg_color='orange', progress_color='gray30')
        self.separator.grid(row=0, column=1, sticky='ns', pady=5)

        self.testing_output_expected_frame = CTkFrame(self.testing_outputs_frame, corner_radius=10, fg_color='transparent')
        self.testing_output_expected_frame.grid(row=0, column=2, sticky='nsew', padx=(0,5), pady=5)

        self.testing_output_expected_label = CTkLabel(self.testing_output_expected_frame, text="Expected output")
        self.testing_output_expected_label.pack(fill='x')

        self.testing_output_expected_value = CTkLabel(self.testing_output_expected_frame, text="0", font=("", 40, 'bold'))
        self.testing_output_expected_value.pack(fill='x')
        
        self.testing_pourcent_label = CTkLabel(self.testing_right_frame, text="Outputs pourcents :", anchor='w')
        self.testing_pourcent_label.pack(padx=15, pady=0, fill='x')

        self.test_pourcent_frame = CTkScrollableFrame(self.testing_right_frame, fg_color='gray25')
        self.test_pourcent_frame.pack(padx=5, pady=5, side='top', expand=True, fill='both')

        self.toggle_entry()

    def train_network(self):
        try:
            epochs_str = self.epochs_entry.get().replace(' ', '')
            epochs = int(epochs_str)
        except ValueError:
            print("Nombre d'epochs invalide.")
            return
        
        self.network.cancel_training = False
        
        self.training_btn.configure(text='Cancel', command=self.cancel_training)
        
        thread = threading.Thread(target=self.train_thread, args=(epochs,), daemon=True)
        thread.start()

    def cancel_training(self):
        self.network.cancel_training = True
        self.training_btn.configure(text='Cancelling...')

    def train_thread(self, epochs):
        def update_ui(progress, epoch, total_epochs, eta):
            def updater():
                self.training_pb.set(progress)
                self.training_pourcent_label.configure(text=f'{int(progress*100)}%')
                self.training_remain_time_label.configure(text=f'Time remain: {int(eta)}s')
            self.after(0, updater)

        original_update = self.network.update_progress
        self.network.update_progress = update_ui
        
        self.network.train_network(self.training_data, epochs)
        
        self.network.update_progress = original_update
        
        if self.save_weights_check_btn.get() == 1:
            self.network.save_weights(f'weights/{epochs}_weights.json')

        self.update_vars()

        def reset_button():
            self.training_btn.configure(text='Train Network', command=self.train_network)
        self.after(0, reset_button)

#network_test_btn
    def test_network(self):
            self.network.load_weights(f'weights/{self.weights_file_menu.get()}')
            test_file = self.test_file_menu.get()
            if not test_file:
                print('Aucun fichier de test sélectionné.')
                return
            filepath = f'data/testing/{test_file}'
            try:
                test_data, input_dim, num_classes = self.network.load_training_data(filepath)
            except Exception as e:
                print('Erreur lors du chargement des données de test:', e)
                return
            
            if self.random_check_btn.get():
                index = random.randint(0, len(test_data) - 1)
            else:
                index_str = self.test_index_entry.get().strip()
                if index_str.isdigit():
                    index = int(index_str)
                    if index < 0 or index >= len(test_data):
                        print("Index de test hors limite. Utilisation d'un index aléatoire.")
                        index = random.randint(0, len(test_data) - 1)
                else:
                    print("Index invalide. Utilisation d'un index aléatoire.")
                    index = random.randint(0, len(test_data) - 1)
            
            sample, expected = test_data[index]
            output = self.network.compute_output(sample)
            
            probs = [(i, float(prob)) for i, prob in enumerate(output)]
            sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
            predicted_digit = sorted_probs[0][0]

            expected_probs = [(i, float(prob)) for i, prob in enumerate(expected)]
            sorted_expected_probs = sorted(expected_probs, key=lambda x: x[1], reverse=True)
            expected_digit = sorted_expected_probs[0][0]

            grid_values = [list(sample[i*8:(i+1)*8]) for i in range(8)]
            canvas = self.digit_canvas
            canvas.delete('all')
            self.draw_rounded_rect(canvas, 0, 0, 496, 496, radius=35, fill='#1d1d1d')
            self.fill_grid(canvas, grid_values, 496, 496, 8, 8, '#1d1d1d', '#dddddd')
            self.draw_grid(canvas, 496, 496, 8, 8, '#dddddd')
            
            self.testing_output_computed_value.configure(text=f'{predicted_digit}')
            self.testing_output_expected_value.configure(text=f'{expected_digit}')
            
            for widget in self.test_pourcent_frame.winfo_children():
                widget.destroy()
            for digit, prob in sorted_probs:
                CTkLabel(self.test_pourcent_frame, text=f'{digit} : {prob*100:.5f}%', font=('Arial', 14)).pack(anchor='w')

    def update_vars(self):
        weights_dir = "weights"
        self.weights_list = [os.path.basename(f) for f in glob.glob(os.path.join(weights_dir, "*.json"))]

        tests_dir = "data/testing"
        self.tests_list = [os.path.basename(f) for f in glob.glob(os.path.join(tests_dir, "*.json"))]
        try:
            self.weights_file_menu.configure(values=self.weights_list)
            self.test_file_menu.configure(values=self.tests_list)
        except :
            pass

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
            self.random_check_btn.focus_set()
            self.test_index_entry.configure(state='disabled')
        else:
            self.test_index_entry.configure(state='normal')