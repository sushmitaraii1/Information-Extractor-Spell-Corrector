import uvicorn
import spacy
from typing import Union
from pattern.text.en import singularize
from autocorrect import Speller

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Information Extraction", version="1.0", description="Testing")

NLP = spacy.load("./models/output/model-best")


class Item(BaseModel):
    paragraph: Union[str, None] = None


# TODO: response model


@app.get("/")
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {"message": "System is healthy"}


@app.post("/information-extractor")
def predict(item: Item):
    data_dict = item.dict()
    para = data_dict["paragraph"]
    doc = NLP(para)
    prof = []
    [
        prof.append(entity.text.capitalize())
        for entity in doc.ents
        if entity.text.capitalize() not in prof
    ]
    res = [singularize(plural) for plural in prof]
    return {"profession": res}


@app.post("/spellcorrector")
def correction(item: Item):
    data_dict = item.dict()
    para = data_dict["paragraph"]
    spell = Speller()
    correct = spell(para)
    return {"corrected sentence": correct}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
