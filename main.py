import customtkinter
from util import read_protein_file, split_based_on_windows, ohe_for_nn, convert_pred_to_str, create_plot
from predictor import PSSPredictor
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ####### Preprocess data and train the neural network #######

    # Preprocess data
    train_seq, train_str = read_protein_file(
        "data/protein-secondary-structure.train")
    test_seq, test_str = read_protein_file(
        "data/protein-secondary-structure.test")

    WINDOW_SIZE = 17

    train_df = split_based_on_windows(train_seq, train_str, WINDOW_SIZE)
    test_df = split_based_on_windows(test_seq, test_str, WINDOW_SIZE)

    X_train, y_train = ohe_for_nn(train_df['sequence'], train_df['string'])
    X_test, y_test = ohe_for_nn(test_df['sequence'], test_df['string'])

    # Train model
    model = PSSPredictor(WINDOW_SIZE)

    # TODO: should we pretrain the model?
    history, loss, accuracy, mae, q3 = model.train(
        X_train, y_train, X_test, y_test, epochs=12)

    ####### CREATE GUI #######

    def on_closing():
        app.withdraw()
        app.quit()

    # Modes: system (default), light, dark
    customtkinter.set_appearance_mode("System")
    # Themes: blue (default), dark-blue, green
    customtkinter.set_default_color_theme("blue")

    app = customtkinter.CTk()  # create CTk window like you do with the Tk window
    app.geometry("1418x600")
    app.title("Predict protein secondary structure with NN")
    app.grid_columnconfigure(1, weight=1)
    app.protocol("WM_DELETE_WINDOW", on_closing)

    # Frames

    frame_col0 = customtkinter.CTkFrame(
        master=app, width=200, height=300, fg_color="transparent")
    frame_col0.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
    frame_text = customtkinter.CTkFrame(master=frame_col0)
    frame_text.grid(row=0, column=0, sticky="nsew")
    label = customtkinter.CTkLabel(
        frame_text, text="Predict protein secondary structure with NN", font=(None, 16))
    label.grid(row=0, column=0, padx=20)

    textbox = customtkinter.CTkTextbox(frame_col0)
    textbox.grid(row=1, column=0, sticky="nsew", pady=20,)

    frame_col1 = customtkinter.CTkFrame(master=app)
    frame_col1.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
    frame_col1.grid_rowconfigure(0, weight=1)

    my_image = customtkinter.CTkImage(light_image=Image.open(
        "images/thumbnail.png"), dark_image=Image.open("images/thumbnail.png"), size=(1000, 200))
    image_label = customtkinter.CTkLabel(
        frame_col1, image=my_image, text="")  # display image with a CTkLabel
    image_label.grid(row=0, column=0)

    frame = customtkinter.CTkFrame(master=app, fg_color="transparent")
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(0, weight=1)

    frame.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
    frame_text = customtkinter.CTkFrame(master=frame)
    frame_text.grid(row=0, column=0, sticky="nsew")
    label = customtkinter.CTkLabel(frame_text, text="Output:", font=(None, 16))
    label.grid(row=0, column=0, padx=20)
    frame_text = customtkinter.CTkFrame(master=frame, fg_color="transparent")
    frame_text.grid_columnconfigure(0, weight=1)
    frame.grid_rowconfigure(0, weight=1)
    frame_text.grid(row=1, column=0, sticky="nsew", pady=20)
    output_label = customtkinter.CTkLabel(
        frame_text, text="None", font=(None, 14), anchor="w")
    output_label.grid(row=0, column=0, padx=20)

    # Predict when user click on button
    def button_function():
        '''
        Function when user clicks a button
        '''
        try:
            error_label.configure(text="")
            text_input = textbox.get('0.0', 'end')
            text_input = text_input.strip()
            text_input = [text_input]
            df = split_based_on_windows(text_input, window_size=WINDOW_SIZE)
            X_data = ohe_for_nn(df['sequence'])
            predictions = model.predict(X_data)
            result = convert_pred_to_str(predictions)
            text_output = "".join(result)
            create_plot(text_input[0], text_output)
            my_image.configure(light_image=Image.open(
                "images/prediction.png"), dark_image=Image.open("images/prediction.png"))
            output_label.configure(text=text_output)
        except ValueError:
            error_label.configure(
                text="There was an error in predicting the secondary structure. Please try again.")

    button = customtkinter.CTkButton(
        master=frame_col0, text="Predict", command=button_function)
    button.grid(row=2, column=0, sticky="nsew")
    error_label = customtkinter.CTkLabel(frame_text, text="", font=(None, 16))
    error_label.grid(row=3, column=0, padx=20)

    # Save training accuracy evolution as an image
    # Placed below tkinter image or else it crashes
    if history:
        plt.figure()
        plt.plot(history.history['q3_score'])
        plt.plot(history.history['val_q3_score'])
        plt.plot([accuracy]*len(history.history['q3_score']), 'g--')
        plt.title('q3 model accuracy')
        plt.ylabel('q3 score accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.savefig('./images/train_accuracy.png')

    app.mainloop()
