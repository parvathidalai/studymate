import os
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from typing import List

class GraniteLLMHandler:
    def __init__(self):
        self.api_key = os.getenv('IBM_API_KEY')
        self.project_id = os.getenv('IBM_PROJECT_ID')
        self.url = os.getenv('IBM_URL')
        self.model_id = os.getenv('GRANITE_MODEL_ID', 'ibm/granite-13b-chat-v2')
        
        # Initialize the model
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize IBM Granite model"""
        try:
            model = Model(
                model_id=self.model_id,
                params={
                    GenParams.DECODING_METHOD: "greedy",
                    GenParams.MAX_NEW_TOKENS: 300,
                    GenParams.TEMPERATURE: 0.5,
                    GenParams.STOP_SEQUENCES: ["\n\n"]
                },
                credentials={
                    "apikey": self.api_key,
                    "url": self.url
                },
                project_id=self.project_id
            )
            return model
        except Exception as e:
            raise Exception(f"Failed to initialize Granite model: {str(e)}")
    
    def generate_answer(self, question: str, context_chunks: List[dict]) -> str:
        """Generate answer using Granite model"""
        try:
            # Construct prompt
            prompt = self._build_prompt(question, context_chunks)
            
            # Generate response
            response = self.model.generate_text(prompt=prompt)
            
            return response.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _build_prompt(self, question: str, context_chunks: List[dict]) -> str:
        """Build prompt for Granite model"""
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        prompt = f"""Based strictly on the following context from academic documents, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear, factual answer based only on the information given in the context above. If the context doesn't contain sufficient information to answer the question, please state that clearly."""

        return prompt