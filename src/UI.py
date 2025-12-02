import gradio as gr
import random

class MockModel:
    def predict(self, input_data):
        count = 0
        if input_data[1] > 60: count += input_data[1]*0.2
        if input_data[2]: count += 1
        if input_data[3]: count += 2
        if input_data[4]: count += 3

        return True if count > 20 else False

    def predict_conf(self):
        return round(random.random(), 2)


model = MockModel()

def predict_lung_cancer(is_male, age, smoking, yellow_fingers, anxiety, peer_pressure,
               chronic_disease, fatigue, allergy, wheezing, coughing,
               shortness_of_breath, swallowing_difficulty, chest_pain) -> list:
    IS_MALE = True if is_male == "Male" else False
    to_boolean = lambda x: True if x == "Yes" else False

    input_data = [
        IS_MALE,
        age,
        to_boolean(smoking),
        to_boolean(yellow_fingers),
        to_boolean(anxiety),
        to_boolean(peer_pressure),
        to_boolean(chronic_disease),
        to_boolean(fatigue),
        to_boolean(allergy),
        to_boolean(wheezing),
        to_boolean(coughing),
        to_boolean(shortness_of_breath),
        to_boolean(swallowing_difficulty),
        to_boolean(chest_pain)
    ]

    # confidence_score = 100
    # return ["You are gay", f"{confidence_score}%"]
    return [model.predict(input_data), f"{model.predict_conf()*100}%"]

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

        with gr.Column():
            gr.Markdown("Result: ")
            result_output = gr.Text(label="Result: ", show_copy_button=True)
            confidence_output = gr.Text(label="Confidence Score: ")

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
            outputs=[result_output, confidence_output]
        )

if __name__ == "__main__":
    demo.launch(share=True)