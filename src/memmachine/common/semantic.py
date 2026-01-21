import spacy
from spacy.tokens import Token

nlp = spacy.load("en_core_web_sm", exclude=["senter", "attribute_ruler", "lemmatizer", "ner"])


def get_semantic_units(text: str) -> list[str]:
    doc = nlp(text)
    doc_semantic_units = []

    SEMANTIC_BREAKS = {"root", "acl", "advcl", "attr", "conj"}

    for sentence in doc.sents:
        head_indexes = {sentence.root.i}
        for token in sentence:
            subtree = list(token.subtree)

            if token.dep_.lower() in SEMANTIC_BREAKS and len(subtree) > 3:
                head_indexes.add(token.i)

        # Assign every token to nearest head.
        head_i_member_tokens_map: dict[int, list[Token]] = {}
        token_i_head_i_map = {}
        for token in sentence:
            curr = token
            while curr.i != curr.head.i and curr.i not in head_indexes:
                curr = curr.head
            token_i_head_i_map[token.i] = curr.i
            head_i_member_tokens_map.setdefault(curr.i, []).append(token)

        # Reconstruct using multi-span joining.
        sentence_units = []
        for head_i, member_tokens in head_i_member_tokens_map.items():
            member_tokens.sort(key=lambda t: t.idx)

            spans = []
            if member_tokens:
                current_start = member_tokens[0]
                current_end = member_tokens[0]

                for i in range(1, len(member_tokens)):
                    next_token = member_tokens[i]

                    is_contiguous = True
                    for intervening_i in range(current_end.i + 1, next_token.i):
                        if token_i_head_i_map.get(intervening_i) != head_i:
                            is_contiguous = False
                            break

                    if is_contiguous:
                        current_end = next_token
                    else:
                        span_text = text[
                            current_start.idx : current_end.idx + len(current_end.text)
                        ]
                        spans.append(span_text)
                        current_start = next_token
                        current_end = next_token

                spans.append(text[current_start.idx: current_end.idx + len(current_end.text)])

            # Join the non-contiguous spans with an ellipsis.
            unit_text = " ... ".join([s.strip() for s in spans if s.strip()])
            sentence_units.append(
                {
                    "start_idx": member_tokens[0].idx,
                    "text": unit_text,
                }
            )

        # Sort by character position to maintain original order.
        sentence_units.sort(key=lambda u: u["start_idx"])
        for unit in sentence_units:
            doc_semantic_units.append(unit["text"])

    return doc_semantic_units
