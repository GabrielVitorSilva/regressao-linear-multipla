import gradio as gr
import joblib
import pandas as pd

# Carregue o modelo
modelo = joblib.load('./modelo_colesterol.pkl')

def predict(grupo_sanguineo, fumante, nivel_atividade_fisica, idade, peso, altura):
    _fumante = 'Sim' if fumante else 'Não'
    predicao_individual = {
        'grupo_sanguineo': [grupo_sanguineo],
        'fumante': [_fumante],
        'nivel_atividade_fisica': [nivel_atividade_fisica],
        'idade': [idade],
        'peso': [peso],
        'altura': [altura],
    }
    predict_df = pd.DataFrame(predicao_individual)
    colesterol = modelo.predict(predict_df)
    return round(colesterol[0], 2)

# Função para validar entradas
def validate_inputs(grupo_sanguineo, fumante, nivel_atividade_fisica, idade, peso, altura):
    if grupo_sanguineo is None or nivel_atividade_fisica is None:
        return "Por favor, preencha todos os campos."
    return predict(grupo_sanguineo, fumante, nivel_atividade_fisica, idade, peso, altura)

demo = gr.Interface(
    fn=validate_inputs,
    inputs=[
        gr.Radio(['O', 'A', 'B', 'AB'], label="Grupo Sanguíneo"),
        gr.Checkbox(label="Fumante"),
        gr.Radio(['Baixo', 'Moderado', 'Alto'], label="Nível de Atividade Física"),
        gr.Slider(20, 80, step=1, label="Idade"),
        gr.Slider(40, 160, step=0.1, label="Peso (kg)"),
        gr.Slider(150, 200, step=1, label="Altura (cm)"),
    ],
    outputs=gr.Textbox(label="Resultado"),
    title="Predição de Colesterol",
    description="Preencha os dados abaixo para predizer o nível de colesterol.",
    theme="default"
)

demo.launch(share=True)
# Rode python app_gradio_colesterol.py