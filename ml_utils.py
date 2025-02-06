import open_clip
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM



class MLutils:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
        print('LLM loaded')
        self.embedding_model, self.embedding_preprocess_train, self.embedding_preprocess_val = open_clip.create_model_and_transforms("RN50", "cc12m")
        self.embedding_model.transformer.eval()
        self.embedding_tokenizer = open_clip.get_tokenizer("RN50")
        print('Embedding models loaded')
        self.device="cpu"


    def get_image_embedding(self,img_fname):
        img = self.encode_img_for_embedding(img_fname)
        return self.embedding_model.encode_image(img.to(device)).detach().numpy().tolist()[0]

    def encode_img_for_embedding(self,img_fname):
        img = Image.open(img_fname)
        img = self.embedding_preprocess_val(img).unsqueeze(0)
        return img

    def get_text_embedding(self,text: str) -> list[float]:
        if not text.strip():
            print("Attempted to get embedding for empty text.")
            return []
        tokens = self.embedding_tokenizer(text)
        return self.embedding_model.encode_text(tokens.squeeze(1).to(self.device)).detach().numpy().tolist()[0]