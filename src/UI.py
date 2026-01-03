import gradio as gr
import pandas as pd
import utils

model_path = "xg_boost_model.pkl"
model = utils.ModelIO.load(model_path)

def predict_lung_cancer(is_male, age, smoking, yellow_fingers, anxiety, peer_pressure,
               chronic_disease, fatigue, allergy, wheezing, coughing,
               shortness_of_breath, swallowing_difficulty, chest_pain) -> list:
    IS_MALE = 1 if is_male == "Male" else 0
    to_int = lambda x: 1 if x == "Yes" else 0

    input_data = [
        IS_MALE,
        age,
        to_int(smoking),
        to_int(yellow_fingers),
        to_int(anxiety),
        to_int(peer_pressure),
        to_int(chronic_disease),
        to_int(fatigue),
        to_int(allergy),
        to_int(wheezing),
        to_int(coughing),
        to_int(shortness_of_breath),
        to_int(swallowing_difficulty),
        to_int(chest_pain)
    ]
    df = pd.DataFrame(input_data).transpose()

    # confidence_score = 100
    # return ["You are gay", f"{confidence_score}%"]
    is_cancer = model.predict(df) == [1]
    output = "You have lung cancer ✨" if is_cancer[0] else "You don't have lung cancer 😔"
    return output

with gr.Blocks() as demo:
    gr.Markdown("# Lung Cancer Detection")
    gr.Markdown("## Fill out the form below to get your detection")

    with gr.Row():
        with gr.Column():
            """NOTE: AGE IS FLOAT HERE"""
            is_male = gr.Radio(choices=["Male", "Female"], label="Gender")
            age = gr.Slider(label="Age", minimum=1.0, maximum=100.0, step=1.0, value=1.0)
            smoking = gr.Radio(choices=["Yes", "No"], label="Do you smoke?")
            yellow_fingers = gr.Radio(choices=["Yes", "No"], label="Do you have yellow fingers?")
            anxiety = gr.Radio(choices=["Yes", "No"], label="Do you have anxiety?")
            peer_pressure = gr.Radio(choices=["Yes", "No"], label="Do you have peer pressure?")
            chronic_disease = gr.Radio(choices=["Yes", "No"], label="Do you have chronic disease?")
            fatigue = gr.Radio(choices=["Yes", "No"], label="Do you often experience fatigue?")
            allergy = gr.Radio(choices=["Yes", "No"], label="Do you have allergy?")
            wheezing = gr.Radio(choices=["Yes", "No"], label="Do you often wheezing?")
            coughing = gr.Radio(choices=["Yes", "No"], label="Do you often cough?")
            shortness_of_breath = gr.Radio(choices=["Yes", "No"], label="Do you have shortness of breath?")
            swallowing_difficulty = gr.Radio(choices=["Yes", "No"], label="Do you have swallowing difficulty?")
            chest_pain = gr.Radio(choices=["Yes", "No"], label="Do you have chest pain?")

            submit_btn = gr.Button("Submit form")

            result_output = gr.Text(label="Result: ", show_copy_button=True)

            gr.Markdown("# Buy me a coffee ☕")

    with gr.Row():
        # QR money transfer
        with gr.Column():
            gr.Image(value="../QR_Phuc.jpg", interactive=False, height=500)
        with gr.Column():
            gr.Image(value="../QR_Qui.jpg", interactive=False, height=500)
        with gr.Column():
            gr.Image(value="../QR_Quoc.jpg", interactive=False, height=500)

        submit_btn.click(
            fn=predict_lung_cancer,
            inputs=[is_male,
                    age,
                    smoking,
                    yellow_fingers,
                    anxiety,
                    peer_pressure,
                    chronic_disease,
                    fatigue,
                    allergy,
                    wheezing,
                    coughing,
                    shortness_of_breath,
                    swallowing_difficulty,
                    chest_pain],
            outputs=result_output
        )

if __name__ == "__main__":
    demo.launch(share=True)