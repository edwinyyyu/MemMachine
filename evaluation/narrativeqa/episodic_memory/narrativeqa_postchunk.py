import argparse
import asyncio
import json
import os
import time

import openai
import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from memmachine.common.utils import async_with

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ANSWER_PROMPT = """
You are asked to answer a question based on your memories.

<instrcutions>
1. Use only the information in your memories to answer the question.
2. Be specific when referring to people, places, objects, events, dates, and concepts.
3. Think carefully about what your memories imply beyond their explicit content.
</instrcutions>

<memories>
{memories}
</memories>

Question: {question}
Your short response to the question without fluff (no more than a sentence):

"""

def get_chunks_with_offsets(content, num_chunks):
    chunk_size = max(len(content) // num_chunks, 400)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=[
            "\n\n\n\n",
            "\n\n\n",
            "\n\n",
            "],\n",
            "},\n",
            "),\n",
            "]\n",
            "}\n",
            ")\n",
            ",\n",
            "\uff1f\n",  # Fullwidth question mark
            "?\n",
            "\uff01\n",  # Fullwidth exclamation mark
            "!\n",
            "\u3002\n",  # Ideographic full stop
            ".\n",
            "\uff1f",  # Fullwidth question mark
            "? ",
            "\uff01",  # Fullwidth exclamation mark
            "! ",
            "\u3002",  # Ideographic full stop
            ". ",
            "; ",
            ": ",
            "—",
            "--",
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            ", ",
            "\u200b",  # Zero-width space
            " ",
            "",
        ],
        keep_separator="end",
    )

    # split_documents provides metadata including offsets in many LangChain versions
    # If using the basic split_text, we track the pointer manually
    chunks = text_splitter.split_text(content)

    chunk_data = []
    pointer = 0
    for chunk in chunks:
        # Since we know the splitter doesn't add characters, only potentially
        # removes them at the absolute end, we find the chunk starting
        # from the current pointer to avoid duplicate issues.
        start = content.find(chunk, pointer)
        end = start + len(chunk)
        chunk_data.append({"text": chunk, "start": start, "end": end})
        pointer = end

    return chunk_data

def get_similar_spans(text_content, query_embedding, num_chunks):
    chunk_objects = get_chunks_with_offsets(text_content, num_chunks)

    if len(chunk_objects) <= 1:
        return [text_content]

    # Vector Similarity
    texts = [c["text"] for c in chunk_objects]
    embeddings = model.encode(texts, convert_to_tensor=True)
    cos_scores = model.similarity(query_embedding, embeddings)[0]

    top_k = min(6, len(chunk_objects))
    top_indices = cos_scores.topk(k=top_k).indices.tolist()

    # Neighbor Expansion
    selected_indices = set()
    for idx in top_indices:
        selected_indices.update({max(0, idx-1), idx, min(len(chunk_objects)-1, idx+1)})

    sorted_indices = sorted(list(selected_indices))
    spans = []

    if sorted_indices:
        # Start the first span
        current_group_start = chunk_objects[sorted_indices[0]]["start"]
        current_group_end = chunk_objects[sorted_indices[0]]["end"]

        for i in range(1, len(sorted_indices)):
            curr_idx = sorted_indices[i]
            prev_idx = sorted_indices[i-1]

            if curr_idx == prev_idx + 1:
                # Continuous: just update the end boundary
                current_group_end = chunk_objects[curr_idx]["end"]
            else:
                # Gap: slice the ORIGINAL text to include all internal separators
                spans.append(text_content[current_group_start:current_group_end])
                current_group_start = chunk_objects[curr_idx]["start"]
                current_group_end = chunk_objects[curr_idx]["end"]

        # Close final span
        spans.append(text_content[current_group_start:current_group_end])

    return spans

def recursive_zoom(text_content, query_embedding, num_chunks=54):
    # Stop if we can't meaningfully split into 400-char blocks
    if len(text_content) <= 400:
        return [text_content]

    spans = get_similar_spans(text_content, query_embedding, num_chunks)

    # Convergence check
    if len(spans) == 1 and spans[0] == text_content:
        return [text_content]

    results = []
    for span in spans:
        # Recursively drill down
        results.extend(recursive_zoom(span, query_embedding, num_chunks))
    return results

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--meta-path", required=True, help="Path to the meta file")
    parser.add_argument("--data-path", required=True, help="Path to the data file")
    parser.add_argument("--qa-path", required=True, help="Path to the qa file")
    parser.add_argument("--target-path", required=True, help="Path to the target file")

    args = parser.parse_args()

    meta_path = args.meta_path
    data_path = args.data_path
    qa_path = args.qa_path
    target_path = args.target_path

    narrativeqa_meta = pd.read_csv(meta_path)
    narrativeqa_qa = pd.read_csv(qa_path)

    openai_client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    async def process_document(row):
        document_id = row["document_id"]

        try:
            with open(f"{data_path}/{document_id}.content", "r") as f:
                document_content = f.read()
        except:
            print(f"Failed to read document {document_id}")
            return None, None

        if not document_content.strip():
            print(f"Empty content for document {document_id}")
            return None, None

        print(
            f"Processing document {document_id}"
        )

        question_results = []

        question_rows = narrativeqa_qa[
            narrativeqa_qa["document_id"] == document_id
        ]
        for _, question_row in question_rows.iterrows():
            question = question_row["question"]
            question_embedding = model.encode(question, convert_to_tensor=True)
            final_spans = recursive_zoom(document_content, question_embedding)

            start_time = time.monotonic()
            formatted_memories = "\n\n".join(final_spans)
            end_time = time.monotonic()
            memory_latency = end_time - start_time

            start_time = time.monotonic()

            response = None
            wait_time_seconds = 1
            while response is None:
                try:
                    response = await openai_client.responses.create(
                        model="gpt-4.1-mini",
                        input=ANSWER_PROMPT.format(
                            memories=formatted_memories,
                            question=question,
                        )
                    )
                except Exception as e:
                    print(f"Error ({document_id}:{question}): {e}", flush=True)
                    await asyncio.sleep(wait_time_seconds)
                    wait_time_seconds = min(wait_time_seconds * 2, 60)

            end_time = time.monotonic()
            llm_latency = end_time - start_time

            response_text = response.output_text

            print(
                f"NQA-Document ID: {document_id}\n"
                f"NQA-Question: {question}\n"
                f"NQA-Answer1: {question_row["answer1"]}\n"
                f"NQA-Answer2: {question_row["answer2"]}\n"
                f"NQA-Response: {response_text}\n"
                f"NQA-Memory latency: {memory_latency:.2f} seconds\n"
                f"NQA-LLM latency: {llm_latency:.2f} seconds\n"
                f"MEMORIES_START\n{formatted_memories}MEMORIES_END\n"
            )

            question_result = {
                "question": question,
                "answer1": question_row["answer1"],
                "answer2": question_row["answer2"],
                "response": response_text,
                "memory_latency": memory_latency,
                "llm_latency": llm_latency,
                "memories": formatted_memories,
            }

            question_results.append(question_result)

        return document_id, question_results

    semaphore = asyncio.Semaphore(20)
    tasks = [
        async_with(semaphore, process_document(row))
        for _, row in narrativeqa_meta[:50].iterrows()]

    results = {
        document_id: question_results
        for document_id, question_results in await asyncio.gather(*tasks) if document_id
    }
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
