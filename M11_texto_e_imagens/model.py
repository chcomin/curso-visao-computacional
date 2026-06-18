from torch import nn
from transformers import pipeline


class TextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Carrega o pipeline do Hugginface, que inclui um tokenizador e
        # um modelo de classificação de texto
        pipe = pipeline(model="distilbert/distilbert-base-cased", task="feature-extraction",
                        device="cpu")
        tokenizer = pipe.tokenizer
        model = pipe.model

        self.tokenizer = tokenizer
        self.model = model
        # id do token associado à classe
        self.cls_token_id = 0
        # Dimensão de saída do modelo distilbert
        self.feature_dim = 768

    def forward(self, text):

        # Se houver uma lista de textos, é preciso preencher com zeros
        # para deixá-los com mesmo tamanho
        padding = isinstance(text, list | tuple)

        tokens = self.tokenizer(text, return_tensors="pt", padding=padding)

        # Envia o texto tokenizado para o mesmo device que o modelo
        tokens = tokens.to(self.model.device)
        res = self.model(**tokens)[0]

        # Acessa os atributos associados com o token de classe
        features = res[:, self.cls_token_id]
        
        return features

