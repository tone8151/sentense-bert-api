from fastapi import FastAPI
from pydantic import BaseModel

from mainbot import MainBot  
from core import SentenceBertJapanese

app = FastAPI()
system = SentenceBertJapanese()

# リクエストbodyを定義
class UserText(BaseModel):
    text: str

@app.get("/")
async def index():
    return "Hello, world"

    
@app.post("/user/")
def create_user(user_text: UserText):
    bot = MainBot(system)
    reply = bot.run(user_text.text)
    return reply