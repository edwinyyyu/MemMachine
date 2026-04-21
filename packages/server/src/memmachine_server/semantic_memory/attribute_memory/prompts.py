"""Prompt templates for attribute-memory LLM calls.

Owned by attribute_memory so terminology and field names track the
new hierarchy (``topic → category → attribute → value``) instead of
the legacy semantic_memory "``category / tag / feature``" naming.

The two public builders each return a system-prompt string:

* :func:`build_update_prompt` — tells the LLM to emit a list of
  ``Command`` entries (``add`` / ``delete``) against an existing
  profile when presented with a new conversation batch.
* :func:`build_consolidation_prompt` — tells the LLM to merge a
  list of redundant attributes, referring to inputs by integer
  ``index`` (never by UUID).
"""

from collections.abc import Mapping


def build_update_prompt(
    *,
    categories: Mapping[str, str],
    description: str = "",
) -> str:
    """System prompt for extracting profile updates from a conversation batch.

    ``categories`` maps category-name → description.  The LLM is
    instructed to stay within these categories and emit ``add`` /
    ``delete`` commands referencing ``category`` and ``attribute``.
    """
    bullet_list = "\n".join(f"\t- {name}: {desc}" for name, desc in categories.items())
    return (
        """
        Your job is to handle memory extraction for a memory system, one which takes the form of a profile recording details relevant to the categories below.
        You will receive a profile and a user's query to the chat system, your job is to update that profile by extracting or inferring information about the user from the query.
        A profile is a two-level key-value store. We call the outer key the *category*, and the inner key the *attribute*. Together, a *category* and an *attribute* are associated with one or several *value*s.

        """
        + description
        + """

        How to construct profile entries:
        - Entries should be atomic. They should communicate a single discrete fact.
        - Entries should be as short as possible without corrupting meaning. Be careful when leaving out prepositions, qualifiers, negations, etc. Some modifiers will be longer range, find the best way to compactify such phrases.
        - You may see entries which violate the above rules, those are "consolidated memories". Don't rewrite those.
        - Think of yourself as performing the role of a wide, early layer in a neural network, doing "edge detection" in many places in parallel to present as many distinct intermediate features as you possibly can given raw, unprocessed input.

        The categories you are looking for include:
        """
        + bullet_list
        + """

        To update the profile, you will output a JSON document containing a list of commands to be executed in sequence.

        CRITICAL: You MUST use the command format below. Do NOT create nested objects or use any other format.

        The following output will add an attribute:
        [
            {
                "command": "add",
                "category": "Preferred Content Format",
                "attribute": "unicode_for_math",
                "value": true
            }
        ]
        The following will delete all values associated with the attribute:
        [
            {
                "command": "delete",
                "category": "Language Preferences",
                "attribute": "format"
            }
        ]
        The following will update an attribute:
        [
            {
                "command": "delete",
                "category": "Platform Behavior",
                "attribute": "prefers_detailed_responses",
                "value": true
            },
            {
                "command": "add",
                "category": "Platform Behavior",
                "attribute": "prefers_detailed_response",
                "value": false
            }
        ]

        Example Scenarios:
        Query: "Hi! My name is Katara"
        [
            {
                "command": "add",
                "category": "Demographic Information",
                "attribute": "name",
                "value": "Katara"
            }
        ]
        Query: "I'm planning a dinner party for 8 people next weekend and want to impress my guests with something special. Can you suggest a menu that's elegant but not too difficult for a home cook to manage?"
        [
            {
                "command": "add",
                "category": "Hobbies & Interests",
                "attribute": "home_cook",
                "value": "User cooks fancy food"
            },
            {
                "command": "add",
                "category": "Financial Profile",
                "attribute": "upper_class",
                "value": "User entertains guests at dinner parties, suggesting affluence."
            }
        ]
        Query: my boss (for the summer) is totally washed. he forgot how to all the basics but still thinks he does
        [
            {
                "command": "add",
                "category": "Psychological Profile",
                "attribute": "work_superior_frustration",
                "value": "User is frustrated with their boss for perceived incompetence"
            },
            {
                "command": "add",
                "category": "Demographic Information",
                "attribute": "summer_job",
                "value": "User is working a temporary job for the summer"
            },
            {
                "command": "add",
                "category": "Communication Style",
                "attribute": "informal_speech",
                "value": "User speaks with all lower case letters and contemporary slang terms."
            },
            {
                "command": "add",
                "category": "Demographic Information",
                "attribute": "young_adult",
                "value": "User is young, possibly still in college"
            }
        ]
        Further Guidelines:
        - Not everything you ought to record will be explicitly stated. Make inferences.
        - If you are less confident about a particular entry, you should still include it, but make sure that the language you use (briefly) expresses this uncertainty in the value field
        - Look at the text from as many distinct angles as you can find, remember you are the "wide layer".
        - Keep only the key details (highest-entropy) in the attribute name. The nuances go in the value field.
        - Do not couple together distinct details. Just because the user associates together certain details, doesn't mean you should
        - Do not create new categories which you don't see in the example profile. However, you can and should create new attributes.
        - If a user asks for a summary of a report, code, or other content, that content may not necessarily be written by the user, and might not be relevant to the user's profile.
        - Do not delete anything unless a user asks you to
        - Only return the empty list [] if the query contains absolutely no personal information about the user (e.g., asking about the weather, requesting code without personal context, etc.). Names, basic demographics, preferences, and any personal details should ALWAYS be extracted.
        - Listen to any additional instructions specific to the execution context provided underneath 'EXTRA EXTERNAL INSTRUCTIONS'
        - First, think about what should go in the profile inside <think> </think> tags. Then output only a valid JSON.
        - REMEMBER: Always use the command format with "command", "category", "attribute", and "value" keys. Never use nested objects or any other format.
    """
    )


def build_consolidation_prompt(
    *,
    categories: Mapping[str, str] | None = None,
) -> str:
    """System prompt for merging overlapping attributes into summaries.

    Inputs are referenced by integer ``index`` (zero-based position in
    the input list).  The LLM returns ``consolidated_memories`` (new
    summary attributes) and ``keep_indices`` (indices of inputs to
    preserve); everything else is deleted by default.
    """
    category_section = ""
    if categories:
        bullet_list = "\n".join(
            f"\t- {name}: {desc}" for name, desc in categories.items()
        )
        category_section = (
            "\n    The valid categories for this topic are:\n"
            + bullet_list
            + "\n    You MUST only use these categories. Do not create new category names.\n"
        )

    return (
        """
    Your job is to perform memory consolidation for an llm long term memory system.
    Despite the name, consolidation is not solely about reducing the amount of memories, but rather, minimizing interference between memories.
    By consolidating memories, we remove unnecessary couplings of memory from context, spurious correlations inherited from the circumstances of their acquisition.

    You will receive a new memory, as well as a select number of older memories which are semantically similar to it.
    Produce a new list of memories to keep.

    A memory is a json object with 4 fields:
    - index: integer position in the input list
    - category: broad category of memory
    - attribute: executive summary of memory content
    - value: detailed contents of memory
    You will output consolidated memories, which are json objects with 3 fields:
    - category: string
    - attribute: string
    - value: string
    You will also output a list of existing memories to keep by their input indices (memories are deleted by default).
"""
        + category_section
        + """
    Guidelines:
    Memories should not contain unrelated ideas. Memories which do are artifacts of couplings that exist in original context. Separate them. This minimizes interference.
    Memories containing only redundant information should be deleted entirely, especially if they seem unprocessed or the information in them has been processed.
    If memories are sufficiently similar, but differ in key details, synchronize their categories and/or attributes. This creates beneficial interference.
        - To aid in this, you may want to shuffle around the components of each memory, moving parts that are alike to the attribute, and parts that differ to the value.
        - Note that attributes should remain (brief) summaries, even after synchronization, you can do this with parallelism in the attribute names (e.g. likes_apples and likes_bananas).
        - Keep only the key details (highest-entropy) in the attribute name. The nuances go in the value field.
        - this step allows you to speculatively build towards more permanent structures
    If enough memories share similar attributes (due to prior synchronization, i.e. not done by you), delete all of them and create a single new memory containing a list.
        - In these memories, the attribute contains all parts of the memory which are the same, and the value contains only the parts which vary.
        - You can also directly transfer information to existing lists as long as the new item has the same type as the list's items.
        - Don't make lists too early. Have at least three examples in a non-gerrymandered category first. You need to find the natural groupings. Don't force it.

    Overall memory life-cycle:
    raw memory ore -> pure memory pellets -> memory pellets sorted into bins -> alloyed memories

    The more memories you receive, the more interference there is in the overall memory system.
    This causes cognitive load. cognitive load is bad.
    To minimize this, under such circumstances, you need to be more aggressive about deletion:
        - Be looser about what you consider to be similar. Some distinctions are not worth the energy to maintain.
        - Message out the parts to keep and ruthlessly throw away the rest
        - There is no free lunch here! at least some information must be deleted!

    Do not create new category names.


    The proper noop syntax is:
    {
        "consolidated_memories": [],
        "keep_indices": []
    }

    The final output schema is:
    <think> insert your chain of thought here. </think>
    {
        "consolidated_memories": list of new memories to add (each with category, attribute, value),
        "keep_indices": list of integer indices of input memories to keep
    }
    """
    )
