import os
import sys
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, LLM_MODEL, TEMPERATURE, MAX_TOKENS

SYSTEM_PROMPT = """You are RuralSathi, an expert AI assistant helping Indian farmers and rural communities. 
You have deep knowledge of Indian agriculture, government welfare schemes, and rural development.

CRITICAL RULES:
1. Answer ONLY using the provided source documents
2. If the sources don't contain the answer, say exactly: "I don't have reliable information about this in my current knowledge base. Please consult your local agricultural officer or Kisan Call Centre (1800-180-1551)."
3. Always cite the source document name at the end of your answer
4. Keep answers practical, clear, and farmer-friendly
5. Use simple language, avoid jargon
6. If mentioning quantities/doses, be precise and repeat what the source says"""

ABSTENTION_TEXT = (
    "I don't have reliable information about this in my current knowledge base. "
    "Please consult your local agricultural officer or Kisan Call Centre (1800-180-1551)."
)


class Generator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Check if .env has it freshly
            from dotenv import load_dotenv
            load_dotenv(override=True)
            self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

        self.model = LLM_MODEL

    def _build_context(self, retrieved_chunks: list) -> str:
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown Document")
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "").strip()
            context_parts.append(f"[Document {i}: {source} (ID: {chunk_id})]\n{text}")
        return "\n\n".join(context_parts)

    def _build_messages(self, query: str, retrieved_chunks: list, conversation_history: list = []) -> list:
        context_str = self._build_context(retrieved_chunks)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Append last 4 turns of conversation history if available
        if conversation_history:
            messages.extend(conversation_history[-4:])

        user_content = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n\n"
            f"User Question: {query}\n\n"
            f"Please answer the question based strictly on the context above and cite the source name."
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, query: str, retrieved_chunks: list, conversation_history: list = []) -> dict:
        if not retrieved_chunks:
            return {
                "answer": ABSTENTION_TEXT,
                "sources": [],
                "model": self.model,
                "tokens_used": 0,
                "abstained": True
            }

        if not self.client:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Please add your key to `.env` as `OPENAI_API_KEY=your_key_here`."
            )

        messages = self._build_messages(query, retrieved_chunks, conversation_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False
        )

        answer_text = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else 0

        sources = [{"name": chunk["source"], "chunk_id": chunk.get("chunk_id", "")} for chunk in retrieved_chunks]

        return {
            "answer": answer_text,
            "sources": sources,
            "model": self.model,
            "tokens_used": tokens_used,
            "abstained": False
        }

    def generate_with_retrieval_check(
        self,
        query: str,
        retrieved_chunks: list,
        is_relevant: bool,
        conversation_history: list = []
    ) -> dict:
        if not is_relevant or not retrieved_chunks:
            return {
                "answer": ABSTENTION_TEXT,
                "sources": [],
                "model": self.model,
                "tokens_used": 0,
                "abstained": True
            }
        return self.generate(query, retrieved_chunks, conversation_history)

    def stream_generate(self, query: str, retrieved_chunks: list, conversation_history: list = []):
        """Yield streaming text chunks for UI."""
        if not retrieved_chunks:
            yield ABSTENTION_TEXT
            return

        if not self.client:
            raise ValueError(
                "OPENAI_API_KEY is not configured. Please add your key to `.env` as `OPENAI_API_KEY=your_key_here`."
            )

        messages = self._build_messages(query, retrieved_chunks, conversation_history)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


if __name__ == "__main__":
    print("=== Testing Generator Module ===")
    sample_chunks = [
        {
            "text": "Apply top dressing of Urea (around 60 kg/ha nitrogen) at the time of first irrigation (Crown Root Initiation stage, 20-25 days after sowing). Ensure adequate soil moisture.",
            "source": "KCC Dataset",
            "chunk_id": "KCC_Dataset_1",
            "metadata": {"crop": "Wheat", "state": "Punjab"}
        },
        {
            "text": "Wheat requires balanced NPK fertilization in the ratio of 120:60:40 kg/ha. Nitrogen should be applied in splits.",
            "source": "icar_wheat_disease_management.txt",
            "chunk_id": "icar_wheat_1",
            "metadata": {"source": "icar_wheat_disease_management.txt"}
        }
    ]

    test_query = "What fertilizer should I apply for wheat crop during crown root initiation?"

    # Check API key presence
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or "your_" in api_key.lower():
        print("\n[NOTE] OPENAI_API_KEY is not yet populated in .env.")
        print("Generator class successfully instantiated and validated.")
        print(f"Context builder test:\n{Generator(api_key='mock_key')._build_context(sample_chunks)[:200]}...")
    else:
        generator = Generator(api_key=api_key)
        res = generator.generate(test_query, sample_chunks)
        print("\nGenerated Response:")
        print(res["answer"])
        print("\nSources Cited:", res["sources"])
        print(f"Tokens Used: {res['tokens_used']} | Abstained: {res['abstained']}")

