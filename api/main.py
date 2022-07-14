import uvicorn
import spacy
from typing import Union, List
from pattern.text.en import singularize
from autocorrect import Speller

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Information Extraction", version="1.0", description="Testing")

NLP = spacy.load("./models/output/model-best")

class Request(BaseModel):
    paragraph: Union[str, None] = None

class ResponseInfoExtractor(BaseModel):
    profession: List[str] = None

class ResposeCorrector(BaseModel):
    correct: Union[str, None] = None

@app.get("/")
def read_home():
    """
    Home endpoint which can be used to test the availability of the application.
    """
    return {"message": "System is healthy"}


@app.post("/information-extractor")
def predict(item: Request):
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
    return ResponseInfoExtractor(profession=res)


@app.post("/spellcorrector")
def correction(item: Request):
    data_dict = item.dict()
    para = data_dict["paragraph"]
    spell = Speller()
    correct = spell(para)
    return ResposeCorrector(correct=correct)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
