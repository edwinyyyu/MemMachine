"""Adversarial test set for the cue-worthiness classifier.

Labels:
  KEEP   - a human would remember this as a useful retrieval cue
  REJECT - contentless conversational mechanics; out of context useless

The classifier MUST never label KEEP as REJECT (false-reject is the costly error).
It is OK if the classifier labels REJECT items as KEEP (false-keep is cheap).
"""

# Curated adversarial test set. Every entry is (label, text).
# Drawn from observed LongMemEval data PLUS adversarial constructions.

KEEP = [
    # --- short factual statements ---
    ("KEEP", "Name is Alice."),
    ("KEEP", "I like trains."),
    ("KEEP", "I'm in Chicago."),
    ("KEEP", "i'm Indian."),
    ("KEEP", "I'll play as half elf"),
    ("KEEP", "My cat's name is Luna."),
    ("KEEP", "I live in Milpitas."),
    ("KEEP", "I'm a vegetarian."),
    ("KEEP", "I drive a Tesla."),
    ("KEEP", "I'm 32 years old."),
    ("KEEP", "I work at Google."),
    ("KEEP", "My anniversary is June 4."),
    ("KEEP", "My son's name is Theo."),
    # --- proper nouns / entities even with formatting ---
    ("KEEP", "Coca-Cola Company"),
    ("KEEP", "**Milpitas, CA:**"),
    ("KEEP", "Seika High School"),
    ("KEEP", "Robert Anton Wilson"),
    ("KEEP", "Telkomsel"),
    ("KEEP", "Karl Marx"),
    ("KEEP", "Pocahontas"),
    ("KEEP", "Leo Lionni"),
    ("KEEP", "Trek Emonda"),
    ("KEEP", "Abus SmartX"),
    ("KEEP", "Akrapovič"),
    ("KEEP", "Brooklyn Nine-Nine"),
    ("KEEP", "The Lumineers and The 1975."),
    ("KEEP", "Summer Vibes"),
    # --- specific concept words / topic terms ---
    ("KEEP", "ahinsa"),
    ("KEEP", "Zen meditation"),
    ("KEEP", "scheduling algorithms"),
    ("KEEP", "FIR and IIR filters"),
    ("KEEP", "Root-Locus plot"),
    ("KEEP", "circular knitting"),
    ("KEEP", "data mining applications"),
    ("KEEP", "Globalization"),
    ("KEEP", "lacrosse trivia"),
    ("KEEP", "Event Model onerror()"),
    # --- specific questions with referents ---
    ("KEEP", "Who did Pocahontas marry?"),
    ("KEEP", "What is Azure Devops?"),
    ("KEEP", "is 81 a prime number"),
    ("KEEP", "Does cupping therapy hurt?"),
    ("KEEP", "Which file is certbot.sh?"),
    ("KEEP", "what is the derivative?"),
    ("KEEP", "How is the payload 240 kg?"),
    ("KEEP", "Is paper coinage?"),
    ("KEEP", "what is dependant variable"),
    ("KEEP", "What did I order at Joe's last week?"),
    # --- preferences / stances ---
    ("KEEP", "conservative"),
    ("KEEP", "I prefer dark mode."),
    ("KEEP", "i want some darker stuff"),
    ("KEEP", "The rich are too highly taxed."),
    # --- story / roleplay content ---
    ("KEEP", "Mustafa comes out of nowhere and attempts to shoot Emaad"),
    ("KEEP", "construct factory"),  # game action, specific
    # --- short answer with concrete value ---
    ("KEEP", "JavaScript"),
    ("KEEP", "Reactjs"),
    ("KEEP", "crow visionaries"),
    ("KEEP", "Whats flowty"),
    # --- non-English concrete content ---
    ("KEEP", "每天開車的時間要怎增加英文能力的方式呢?"),
    # --- short events / dates ---
    (
        "KEEP",
        "I got the new pair of Adidas running shoes on February 12th, a Thursday.",
    ),
]

REJECT = [
    # --- pure acks ---
    ("REJECT", "Acknowledged."),
    ("REJECT", "acknowledged"),
    ("REJECT", "Memorized."),
    ("REJECT", "Okay."),
    ("REJECT", "ok."),
    ("REJECT", "Ok"),
    ("REJECT", "Yes."),
    ("REJECT", "No."),
    ("REJECT", "yes"),
    ("REJECT", "no"),
    ("REJECT", "Yes, I understand."),
    ("REJECT", "Understood. Please go ahead."),
    ("REJECT", "Yup"),
    ("REJECT", "Got it."),
    ("REJECT", "Sure."),
    ("REJECT", "Strongly disagree."),
    ("REJECT", "Agree."),
    ("REJECT", "Disagree."),
    # --- greetings / sign-offs ---
    ("REJECT", "hi"),
    ("REJECT", "Hi"),
    ("REJECT", "Hello"),
    ("REJECT", "hello"),
    ("REJECT", "Hi there!"),
    ("REJECT", "Hello! How can I assist you today?"),
    ("REJECT", "Hello! How can I help you today?"),
    ("REJECT", "How can I assist you today?"),
    ("REJECT", "Hi chat -"),
    ("REJECT", "Yo yo yo yo"),
    ("REJECT", "Goodbye! Have a great day!"),
    ("REJECT", "Hi there! It's great to see you! What's up?"),
    ("REJECT", "Hey there, how can I help?"),
    # --- generic continue / more ---
    ("REJECT", "continue"),
    ("REJECT", "Continue"),
    ("REJECT", "Please continue"),
    ("REJECT", "please continue"),
    ("REJECT", "continue please"),
    ("REJECT", "continue:"),
    ("REJECT", "Continue writing please"),
    ("REJECT", "Continue writing please\n\nPlease write in English language."),
    ("REJECT", "continue\n \n \n \n 지금 번역하기"),
    ("REJECT", "Coninue the story"),
    ("REJECT", "continue writing"),
    ("REJECT", "more"),
    ("REJECT", "More"),
    ("REJECT", "more please"),
    ("REJECT", "more questions"),
    ("REJECT", "list more"),
    ("REJECT", "are there more?"),
    ("REJECT", "what are more"),
    ("REJECT", "next"),
    ("REJECT", "keep going"),
    ("REJECT", "Regenerate"),
    ("REJECT", "moreShare Prompt"),
    ("REJECT", "finish please"),
    ("REJECT", "finish your answer"),
    ("REJECT", "finish itShare Prompt"),
    # --- generic instructions w/o referent ---
    ("REJECT", "Make it longer"),
    ("REJECT", "make it UWU"),
    ("REJECT", "Reduce the length by half"),
    ("REJECT", "Reformat into a code block"),
    ("REJECT", "Put this in tables"),
    ("REJECT", "Show the whole table"),
    ("REJECT", "Continue table"),
    ("REJECT", "Expand on this"),
    ("REJECT", "Expand the code above further"),
    ("REJECT", "simplify even further"),
    ("REJECT", "give me five more like this"),
    ("REJECT", "Give me another Answer"),
    ("REJECT", "Do another one"),
    ("REJECT", "Another."),
    ("REJECT", "any other ideas?"),
    ("REJECT", "can you send the rest?"),
    ("REJECT", "What is the second part?"),
    ("REJECT", "Can you be more specific?"),
    ("REJECT", "What did you remove or change?"),
    ("REJECT", "present 5 more concept"),
    ("REJECT", "Write another scene"),
    ("REJECT", "Do you have any other ideas?"),
    # --- formatting fragments ---
    ("REJECT", "**Resources:**"),
    ("REJECT", "**Additional Tips:**"),
    ("REJECT", "**Tips:**"),
    ("REJECT", "**Challenges:**"),
    ("REJECT", "**Attractions:**"),
    ("REJECT", "**Brands:**"),
    ("REJECT", "**Instructions:**"),
    ("REJECT", "1.Instructions:"),
    ("REJECT", "Instructions (suite):Instructions:"),
    ("REJECT", "---"),
    ("REJECT", "```"),
    ("REJECT", "4."),
    ("REJECT", "VI. B."),
    ("REJECT", "VIII. C."),
    ("REJECT", "more pages"),
    ("REJECT", "​"),  # zero-width
    ("REJECT", "|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |"),
    (
        "REJECT",
        "| Component | User Action |\n| --- | --| Component | User Action |\n| --- | --",
    ),
    # --- generic affect / thanks ---
    ("REJECT", "Thanks!"),
    ("REJECT", "Thank you"),
    ("REJECT", "Thank you."),
    ("REJECT", "thank you"),
    ("REJECT", "Thanks"),
    ("REJECT", "Thank you, happy to help!"),
    ("REJECT", "Excellent!"),
    ("REJECT", "Perfect"),
    ("REJECT", "This is great"),
    ("REJECT", "That's really helpful, thanks!"),
    ("REJECT", "I hope this helps!"),
    ("REJECT", "I'm glad I could help!"),
    ("REJECT", "I'd be happy to help you with that."),
    ("REJECT", "I'm happy to help!"),
    ("REJECT", "You're welcome! Let me know if you have any other questions."),
    (
        "REJECT",
        "I hope this is more helpful. Let me know if you have any further questions or clarifications needed.",
    ),
    ("REJECT", "What else can I help you with?"),
    ("REJECT", "Yeah that's better."),
    ("REJECT", "Okay I understand that."),
    ("REJECT", "Okay, then it's not arbitrary"),
    ("REJECT", "I just did"),
    ("REJECT", "I can do that, GPT."),
    # --- placeholders / nonsense ---
    ("REJECT", "test"),
    ("REJECT", "spq"),
    ("REJECT", "read"),
    ("REJECT", "Too many requests in 1 hour. Try again later."),
    # --- single character / number ---
    ("REJECT", "C"),
    ("REJECT", "B"),
    ("REJECT", "1"),
    ("REJECT", "2"),
    ("REJECT", "7"),
    ("REJECT", "You and"),
    # --- generic agent boilerplate ---
    ("REJECT", "Please respond as the user."),
    ("REJECT", "Please respond with the next message from the user."),
    # --- vague follow-ups w/o referent ---
    ("REJECT", "Yeah that's better."),
    ("REJECT", "Its not the end"),
    ("REJECT", "ITS NOT THE END"),
    ("REJECT", "Analyse the text:"),
    ("REJECT", "give a specific example of each"),
    ("REJECT", "give me 2 others content"),
    ("REJECT", "give me 2 others better content"),
    ("REJECT", "please do so"),
    ("REJECT", "Wait, I think I need to ask something else."),
    ("REJECT", "What's on your mind?"),
    # --- bare numeric values w/ no entity at all ---
    # ("$45" alone is contentless: no noun, no referent, just a price)
    ("REJECT", "$45"),
]


# Items where reasonable humans would disagree — held out of scoring.
# Under the asymmetric rule the classifier should default to KEEP on these,
# but we don't penalize either choice.
BORDERLINE = [
    "32 hours a week",
    "That's my birthday.",
    "Yes, the movie is a cartoon.",
    "not sure... maybe around the 20th century",
    "something to do with politics",
    "crayon",
    "Ты кто?",
    "shorter, leave premium",
    "any math or coding joke?",
    "{she thought for a moment}",  # roleplay stage direction, minimal info
    '"im hungreeeee"',  # quoted character speech, ambiguous
    "March 11th.",  # bare date, no anchor noun
]


def all_items():
    return KEEP + REJECT


if __name__ == "__main__":
    print(f"KEEP: {len(KEEP)}")
    print(f"REJECT: {len(REJECT)}")
    print(f"TOTAL: {len(KEEP) + len(REJECT)}")
