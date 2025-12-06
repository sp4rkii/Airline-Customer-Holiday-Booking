import os
from typing import Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

def get_llm(model_name: str):
    """
    Factory function to return the requested LLM instance.
    """
    if model_name == "Gemini Flash":
        if "GOOGLE_API_KEY" not in os.environ:
             raise ValueError("GOOGLE_API_KEY not found in environment.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
    
    elif model_name == "Mistral-7B":
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment. Please add it to your .env file.")
        
        # Use ChatHuggingFace for conversational models
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        return ChatHuggingFace(llm=llm)

    elif model_name == "Zephyr-7B":
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment. Please add it to your .env file.")
        
        # Use ChatHuggingFace for conversational models
        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        return ChatHuggingFace(llm=llm)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
