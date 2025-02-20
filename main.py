from network import NeuralNetwork
from interface import Interface
from customtkinter import CTkLabel
import threading
import random
import numpy as np

class Diane:
    def __init__(self):
        self.network = NeuralNetwork(architecture=[64, 16, 10], learning_rate=0.1)
        self.network.cancel_training = False
        self.training_data, self.input_dim, self.num_classes = self.network.load_training_data('data/training/training_data.json')
        
        self.interface = Interface(self.network, self.training_data, self.input_dim, self.num_classes)
        
        self.setup_callbacks()
        
        self.interface.mainloop()

    def setup_callbacks(self):
        self.interface.training_btn.configure(text='Train Network', command=self.train_network)
        self.interface.random_test_btn.configure(command=self.test_network)
        self.interface.set_weights_file_btn.configure(command=self.set_weights)

    def train_network(self):
        try:
            epochs_str = self.interface.epochs_entry.get().replace(' ', '')
            epochs = int(epochs_str)
        except ValueError:
            print("Nombre d'epochs invalide.")
            return
        
        self.network.cancel_training = False
        
        self.interface.training_btn.configure(text='Cancel', command=self.cancel_training)
        
        thread = threading.Thread(target=self.train_thread, args=(epochs,), daemon=True)
        thread.start()

    def cancel_training(self):
        self.network.cancel_training = True
        self.interface.training_btn.configure(text='Cancelling...')

    def train_thread(self, epochs):
        def update_ui(progress, epoch, total_epochs, eta):
            def updater():
                self.interface.training_pb.set(progress)
                self.interface.training_pourcent_label.configure(text=f'{int(progress*100)}%')
                self.interface.training_remain_time_label.configure(text=f'Time remain: {int(eta)}s')
            self.interface.after(0, updater)

        original_update = self.network.update_progress
        self.network.update_progress = update_ui
        
        self.network.train_network(self.training_data, epochs)
        
        self.network.update_progress = original_update
        
        def reset_button():
            self.interface.training_btn.configure(text='Train Network', command=self.train_network)
        self.interface.after(0, reset_button)

    def test_network(self):
        test_file = self.interface.test_file_menu.get()
        if not test_file:
            print('Aucun fichier de test sélectionné.')
            return
        filepath = f'data/testing/{test_file}'
        try:
            test_data, input_dim, num_classes = self.network.load_training_data(filepath)
        except Exception as e:
            print('Erreur lors du chargement des données de test:', e)
            return
        
        if self.interface.random_check_btn.get():
            index = random.randint(0, len(test_data) - 1)
        else:
            index_str = self.interface.test_intex_entry.get().strip()
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
        
        grid_values = [list(sample[i*8:(i+1)*8]) for i in range(8)]
        canvas = self.interface.digit_canvas
        canvas.delete('all')
        self.interface.draw_rounded_rect(canvas, 0, 0, 496, 496, radius=35, fill='#1d1d1d')
        self.interface.fill_grid(canvas, grid_values, 496, 496, 8, 8, '#1d1d1d', '#dddddd')
        self.interface.draw_grid(canvas, 496, 496, 8, 8, '#dddddd')
        
        for widget in self.interface.test_output_frame.winfo_children():
            widget.destroy()
        predicted_label = CTkLabel(self.interface.test_output_frame, text=str(predicted_digit), font=('Arial', 48))
        predicted_label.pack(expand=True)
        
        for widget in self.interface.test_pourcent_frame.winfo_children():
            widget.destroy()
        for digit, prob in sorted_probs:
            CTkLabel(self.interface.test_pourcent_frame, text=f'{digit} : {prob*100:.1f}%', font=('Arial', 14)).pack(anchor='w')
        
        print(f"Test effectué sur l'échantillon index {index}. Chiffre prédit : {predicted_digit}")

    def set_weights(self):
        weights_filename = self.interface.weights_file_menu.get()
        if weights_filename:
            self.network.load_weights(f'weights/{weights_filename}')
            print(f'Poids chargés depuis : weights/{weights_filename}')
        else:
            print('Aucun fichier de poids sélectionné.')

if __name__ == '__main__':
    Diane()