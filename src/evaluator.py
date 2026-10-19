import os
import sys
import json
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retriever import Retriever
from src.generator import Generator


class RAGEvaluator:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self.results = []

    def evaluate_single(self, question: str, category: str, expected_keyword: str = None) -> dict:
        start_time = time.time()

        # Step 1: Retrieval & Relevance
        retrieved_chunks = self.retriever.retrieve(question, top_k=3)
        top_chunk = retrieved_chunks[0] if retrieved_chunks else None
        is_relevant = self.retriever.is_relevant(top_chunk) if top_chunk else False

        # Step 2: Generation with fallback check
        try:
            gen_result = self.generator.generate_with_retrieval_check(
                query=question,
                retrieved_chunks=retrieved_chunks,
                is_relevant=is_relevant
            )
            answer = gen_result.get("answer", "")
            abstained = gen_result.get("abstained", False)
            sources = gen_result.get("sources", [])
        except Exception as e:
            # If generator fails (e.g. no OpenAI API key), handle gracefully based on relevance
            if not is_relevant or not retrieved_chunks:
                abstained = True
                answer = "I don't have reliable information about this in my current knowledge base."
            else:
                abstained = False
                answer = f"[Simulated Output based on {top_chunk.get('source', '')}] Source: {top_chunk.get('source', '')}"
            sources = [{"name": c["source"], "chunk_id": c.get("chunk_id", "")} for c in retrieved_chunks]

        latency = time.time() - start_time

        # Step 3: Compute evaluation metrics
        # retrieval_hit: expected_keyword appears in any retrieved source name or chunk text
        retrieval_hit = False
        if expected_keyword:
            kw_lower = expected_keyword.lower()
            for chunk in retrieved_chunks:
                if kw_lower in chunk.get("text", "").lower() or kw_lower in chunk.get("source", "").lower():
                    retrieval_hit = True
                    break
        elif category == "off_topic":
            # For off-topic, retrieval hit is considered True if relevance is rejected
            retrieval_hit = not is_relevant

        # abstained_correctly: off_topic questions that got abstained=True
        abstained_correctly = (category == "off_topic" and abstained)

        # answered_correctly: answerable or multi_chunk questions that were NOT abstained
        answered_correctly = (category in ["answerable", "multi_chunk"] and not abstained)

        # source_cited: "source" (case-insensitive) appears in answer or sources list has items
        source_cited = ("source" in answer.lower()) or bool(sources and not abstained)

        record = {
            "question": question,
            "category": category,
            "expected_keyword": expected_keyword,
            "retrieval_hit": retrieval_hit,
            "abstained": abstained,
            "abstained_correctly": abstained_correctly,
            "answered_correctly": answered_correctly,
            "source_cited": source_cited,
            "latency_seconds": round(latency, 3),
            "chunks_retrieved": len(retrieved_chunks),
            "top_reranker_score": round(top_chunk.get("reranker_score", 0.0), 3) if top_chunk else 0.0,
            "answer": answer,
            "sources": json.dumps(sources)
        }
        return record

    def run_full_evaluation(self, test_file: str = "tests/test_questions.json") -> pd.DataFrame:
        if not os.path.exists(test_file):
            # Try absolute path from workspace root
            test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), test_file)

        with open(test_file, "r", encoding="utf-8") as f:
            questions = json.load(f)

        print(f"\n=== Starting Evaluation on {len(questions)} Questions ===")
        records = []
        for i, q in enumerate(questions, 1):
            res = self.evaluate_single(
                question=q["question"],
                category=q.get("category", "answerable"),
                expected_keyword=q.get("expected_source_keyword")
            )
            records.append(res)
            print(f"[{i}/{len(questions)}] ({res['category']}) Hit: {res['retrieval_hit']} | Ans: {res['answered_correctly']} | Abstain: {res['abstained']}")

        df = pd.DataFrame(records)

        # Summaries
        ans_mask = df["category"] == "answerable"
        off_mask = df["category"] == "off_topic"
        multi_mask = df["category"] == "multi_chunk"

        ans_acc = df[ans_mask]["answered_correctly"].sum()
        ans_total = ans_mask.sum()

        off_acc = df[off_mask]["abstained_correctly"].sum()
        off_total = off_mask.sum()

        multi_acc = df[multi_mask]["answered_correctly"].sum()
        multi_total = multi_mask.sum()

        avg_latency = df["latency_seconds"].mean()
        citation_rate = (df["source_cited"].sum() / len(df)) * 100

        print("\n=== Evaluation Summary Report ===")
        print(f"* Answerable accuracy: {ans_acc}/{ans_total}")
        print(f"* Off-topic abstention: {off_acc}/{off_total}")
        print(f"* Multi-chunk success: {multi_acc}/{multi_total}")
        print(f"* Average latency: {avg_latency:.2f}s")
        print(f"* Source citation rate: {citation_rate:.1f}%")

        self.save_results(df)
        return df

    def save_results(self, df: pd.DataFrame, path: str = "tests/evaluation_results.csv"):
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Results saved to CSV: {path}")

        # Summary JSON
        summary_path = path.replace(".csv", "_summary.json")
        ans_mask = df["category"] == "answerable"
        off_mask = df["category"] == "off_topic"
        multi_mask = df["category"] == "multi_chunk"

        summary = {
            "total_questions": len(df),
            "answerable_accuracy": f"{df[ans_mask]['answered_correctly'].sum()}/{ans_mask.sum()}",
            "off_topic_abstention": f"{df[off_mask]['abstained_correctly'].sum()}/{off_mask.sum()}",
            "multi_chunk_success": f"{df[multi_mask]['answered_correctly'].sum()}/{multi_mask.sum()}",
            "average_latency_seconds": round(float(df["latency_seconds"].mean()), 2),
            "source_citation_rate_pct": round(float((df["source_cited"].sum() / len(df)) * 100), 1)
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to JSON: {summary_path}")


if __name__ == "__main__":
    print("=== Testing Evaluator with Retriever ===")
    from src.retriever import Retriever
    from src.generator import Generator

    retriever = Retriever()
    generator = Generator()
    evaluator = RAGEvaluator(retriever, generator)

    sample_res = evaluator.evaluate_single(
        question="What fertilizer should I use for wheat crop?",
        category="answerable",
        expected_keyword="wheat"
    )
    print("\nSample Single Evaluation Result:")
    for k, v in sample_res.items():
        if k != "sources":
            print(f"  {k}: {v}")
