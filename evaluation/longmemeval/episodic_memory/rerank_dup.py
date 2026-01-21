import argparse
import asyncio
import os
from datetime import datetime
from uuid import uuid4

import boto3
import neo4j
import openai
from dotenv import load_dotenv
from longmemeval_models import (
    LongMemEvalItem,
    load_longmemeval_dataset,
)

from memmachine.common.embedder.openai_embedder import (
    OpenAIEmbedder,
    OpenAIEmbedderParams,
)
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine.common.reranker.amazon_bedrock_reranker import (
    AmazonBedrockReranker,
    AmazonBedrockRerankerParams,
)
from memmachine.common.utils import async_with
from memmachine.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine.episodic_memory.declarative_memory import (
    ContentType,
    DeclarativeMemory,
    DeclarativeMemoryParams,
    Episode,
)


async def main():
    region = "us-west-2"
    aws_client = boto3.client(
        "bedrock-agent-runtime",
        region_name=region,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    reranker = AmazonBedrockReranker(
        AmazonBedrockRerankerParams(
            client=aws_client,
            region=region,
            model_id="cohere.rerank-v3-5:0",
        )
    )

    print(
        await reranker.score(
            "Where did Caroline move from 4 years ago?",
            [
                '''Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"''',
                '''Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"\nCaroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"''',
                '''Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"\nCaroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"\nCaroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"''',
                '''Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."''',
                '''Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."\nCaroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."''',
                '''Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."\nCaroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."\nCaroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."\nCaroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."''',
                """

Engineering work can actually span both the **services** and **goods** sides of the economy, depending on what exactly is being produced. Let me break it down clearly:

1. **Engineering as a service:**

   * When engineers provide **consulting, design, planning, or problem-solving** for clients, it’s considered a service.
   * Examples:

     * Designing a bridge or a building
     * Providing software engineering solutions
     * Engineering consulting for a manufacturing process
   * These are intangible outputs—you’re paying for expertise, knowledge, and intellectual labor rather than a physical product.

2. **Engineering as part of goods production:**

   * When engineers are directly involved in **creating physical products**, their work contributes to the goods economy.
   * Examples:

     * Designing and building machinery
     * Developing a new consumer electronic device
     * Engineering parts for cars, planes, or industrial equipment
   * Here, engineering is tied to the production of tangible goods.

✅ **Bottom line:** Engineering itself is often **a service**, because it’s the application of expertise, but it is also **closely linked to goods production** when that expertise results in a physical product. In economic classification, most engineering is counted under the **service sector**, even if it supports manufacturing.

If you want, I can make a **simple diagram showing engineering in services vs goods economy**—it’s easier to see visually. Do you want me to do that?
In basic biological classification, there are **five main classes of vertebrates**:

1. **Fish** – cold-blooded, live in water, usually have scales and fins
2. **Amphibians** – cold-blooded, live part of their life in water and part on land (e.g., frogs)
3. **Reptiles** – cold-blooded, mostly land-dwelling, have dry scaly skin (e.g., snakes, lizards)
4. **Birds (Aves)** – warm-blooded, have feathers and lay eggs
5. **Mammals** – warm-blooded, have hair or fur, and produce milk for their young

👉 **Note:** In more advanced taxonomy, fish are sometimes split into multiple classes (such as jawless, cartilaginous, and bony fish), which increases the total number. But at the school level, the answer is **five classes**.
Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"

Ah, this is a really interesting question because it’s deceptively tricky. On the surface, it might seem like “anyone can make music” these days—smartphones, DAWs (digital audio workstations), and AI-assisted tools have made production easier than ever. But becoming famous and making a sustainable income is still **much harder**. Let me break it down carefully:

---

### 1. **Making music is easy, but standing out is hard**

* **Anyone can create a song**, but **millions of songs are uploaded every day**. Fame comes from being noticed, not just producing content.
* Musicians who break through often have **a unique voice, style, or story** that connects emotionally with an audience. Even technically perfect songs can go unnoticed if they don’t resonate.

---

### 2. **Distribution and exposure matter**

* Platforms like Spotify, YouTube, TikTok, and Instagram let anyone release music—but **attention is limited**.
* Going viral often depends on timing, trends, algorithmic favor, and luck, not just quality. Many excellent songs never reach wide audiences.
* Collaborations with other artists, influencers, or playlists can amplify reach.

---

### 3. **Building a personal brand**

* Fame isn’t just about songs—it’s about **identity**. Fans follow people, not just tracks.
* Musicians often become known for **their personality, image, or lifestyle**, which is shared across social media and media appearances.
* Think of Billie Eilish’s distinctive style or Bad Bunny’s social media presence—they’re as recognizable for their persona as their music.

---

### 4. **Monetization is multi-layered**

Even if music creation is easy, **making money requires multiple revenue streams**:

1. **Streaming royalties**: Usually small per stream. Millions of streams are needed to earn a decent living.
2. **Live performances**: Tours, concerts, and festivals can bring in the bulk of income.
3. **Merchandise**: Clothing, posters, limited edition items.
4. **Sponsorships and brand deals**: Popular musicians partner with brands for income.
5. **Songwriting/production credits**: Other artists or media using your music can generate royalties.
6. **Sync licensing**: Music in TV, movies, or ads pays well if your song gets picked.

---

### 5. **Networking and industry knowledge**

* Connections with producers, managers, record labels, and other artists still matter. Even with digital platforms, **industry support accelerates exposure**.
* Musicians who treat it like a business—understanding contracts, promotion, and analytics—tend to survive and thrive longer.

---

### 6. **Consistency and persistence**

* Fame is rarely instant. Even artists who “explode” often spent **years honing skills, building a fanbase, and releasing content**.
* Consistent output and engagement with fans keep an artist visible in a crowded market.

---

✅ **In short:**
Easy access to tools has lowered the barrier to entry, but **fame and income come from a combination of talent, uniqueness, marketing, networking, and sometimes luck**. Music creation is just one small piece of the puzzle.

---

If you want, I can make a **step-by-step blueprint showing how an independent musician today could realistically become famous and earn money**, including timelines and practical actions. That might make this much clearer.

Do you want me to do that?
""",
                """

Engineering work can actually span both the **services** and **goods** sides of the economy, depending on what exactly is being produced. Let me break it down clearly:

1. **Engineering as a service:**

   * When engineers provide **consulting, design, planning, or problem-solving** for clients, it’s considered a service.
   * Examples:

     * Designing a bridge or a building
     * Providing software engineering solutions
     * Engineering consulting for a manufacturing process
   * These are intangible outputs—you’re paying for expertise, knowledge, and intellectual labor rather than a physical product.

2. **Engineering as part of goods production:**

   * When engineers are directly involved in **creating physical products**, their work contributes to the goods economy.
   * Examples:

     * Designing and building machinery
     * Developing a new consumer electronic device
     * Engineering parts for cars, planes, or industrial equipment
   * Here, engineering is tied to the production of tangible goods.

✅ **Bottom line:** Engineering itself is often **a service**, because it’s the application of expertise, but it is also **closely linked to goods production** when that expertise results in a physical product. In economic classification, most engineering is counted under the **service sector**, even if it supports manufacturing.

If you want, I can make a **simple diagram showing engineering in services vs goods economy**—it’s easier to see visually. Do you want me to do that?
In basic biological classification, there are **five main classes of vertebrates**:

1. **Fish** – cold-blooded, live in water, usually have scales and fins
2. **Amphibians** – cold-blooded, live part of their life in water and part on land (e.g., frogs)
3. **Reptiles** – cold-blooded, mostly land-dwelling, have dry scaly skin (e.g., snakes, lizards)
4. **Birds (Aves)** – warm-blooded, have feathers and lay eggs
5. **Mammals** – warm-blooded, have hair or fur, and produce milk for their young

👉 **Note:** In more advanced taxonomy, fish are sometimes split into multiple classes (such as jawless, cartilaginous, and bony fish), which increases the total number. But at the school level, the answer is **five classes**.
Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."

Ah, this is a really interesting question because it’s deceptively tricky. On the surface, it might seem like “anyone can make music” these days—smartphones, DAWs (digital audio workstations), and AI-assisted tools have made production easier than ever. But becoming famous and making a sustainable income is still **much harder**. Let me break it down carefully:

---

### 1. **Making music is easy, but standing out is hard**

* **Anyone can create a song**, but **millions of songs are uploaded every day**. Fame comes from being noticed, not just producing content.
* Musicians who break through often have **a unique voice, style, or story** that connects emotionally with an audience. Even technically perfect songs can go unnoticed if they don’t resonate.

---

### 2. **Distribution and exposure matter**

* Platforms like Spotify, YouTube, TikTok, and Instagram let anyone release music—but **attention is limited**.
* Going viral often depends on timing, trends, algorithmic favor, and luck, not just quality. Many excellent songs never reach wide audiences.
* Collaborations with other artists, influencers, or playlists can amplify reach.

---

### 3. **Building a personal brand**

* Fame isn’t just about songs—it’s about **identity**. Fans follow people, not just tracks.
* Musicians often become known for **their personality, image, or lifestyle**, which is shared across social media and media appearances.
* Think of Billie Eilish’s distinctive style or Bad Bunny’s social media presence—they’re as recognizable for their persona as their music.

---

### 4. **Monetization is multi-layered**

Even if music creation is easy, **making money requires multiple revenue streams**:

1. **Streaming royalties**: Usually small per stream. Millions of streams are needed to earn a decent living.
2. **Live performances**: Tours, concerts, and festivals can bring in the bulk of income.
3. **Merchandise**: Clothing, posters, limited edition items.
4. **Sponsorships and brand deals**: Popular musicians partner with brands for income.
5. **Songwriting/production credits**: Other artists or media using your music can generate royalties.
6. **Sync licensing**: Music in TV, movies, or ads pays well if your song gets picked.

---

### 5. **Networking and industry knowledge**

* Connections with producers, managers, record labels, and other artists still matter. Even with digital platforms, **industry support accelerates exposure**.
* Musicians who treat it like a business—understanding contracts, promotion, and analytics—tend to survive and thrive longer.

---

### 6. **Consistency and persistence**

* Fame is rarely instant. Even artists who “explode” often spent **years honing skills, building a fanbase, and releasing content**.
* Consistent output and engagement with fans keep an artist visible in a crowded market.

---

✅ **In short:**
Easy access to tools has lowered the barrier to entry, but **fame and income come from a combination of talent, uniqueness, marketing, networking, and sometimes luck**. Music creation is just one small piece of the puzzle.

---

If you want, I can make a **step-by-step blueprint showing how an independent musician today could realistically become famous and earn money**, including timelines and practical actions. That might make this much clearer.

Do you want me to do that?
""",
                """

Engineering work can actually span both the **services** and **goods** sides of the economy, depending on what exactly is being produced. Let me break it down clearly:

1. **Engineering as a service:**

   * When engineers provide **consulting, design, planning, or problem-solving** for clients, it’s considered a service.
   * Examples:

     * Designing a bridge or a building
     * Providing software engineering solutions
     * Engineering consulting for a manufacturing process
   * These are intangible outputs—you’re paying for expertise, knowledge, and intellectual labor rather than a physical product.

2. **Engineering as part of goods production:**

   * When engineers are directly involved in **creating physical products**, their work contributes to the goods economy.
   * Examples:

     * Designing and building machinery
     * Developing a new consumer electronic device
     * Engineering parts for cars, planes, or industrial equipment
   * Here, engineering is tied to the production of tangible goods.

✅ **Bottom line:** Engineering itself is often **a service**, because it’s the application of expertise, but it is also **closely linked to goods production** when that expertise results in a physical product. In economic classification, most engineering is counted under the **service sector**, even if it supports manufacturing.

If you want, I can make a **simple diagram showing engineering in services vs goods economy**—it’s easier to see visually. Do you want me to do that?
In basic biological classification, there are **five main classes of vertebrates**:

1. **Fish** – cold-blooded, live in water, usually have scales and fins
2. **Amphibians** – cold-blooded, live part of their life in water and part on land (e.g., frogs)
3. **Reptiles** – cold-blooded, mostly land-dwelling, have dry scaly skin (e.g., snakes, lizards)
4. **Birds (Aves)** – warm-blooded, have feathers and lay eggs
5. **Mammals** – warm-blooded, have hair or fur, and produce milk for their young

👉 **Note:** In more advanced taxonomy, fish are sometimes split into multiple classes (such as jawless, cartilaginous, and bony fish), which increases the total number. But at the school level, the answer is **five classes**.

Ah, this is a really interesting question because it’s deceptively tricky. On the surface, it might seem like “anyone can make music” these days—smartphones, DAWs (digital audio workstations), and AI-assisted tools have made production easier than ever. But becoming famous and making a sustainable income is still **much harder**. Let me break it down carefully:
Caroline

---

### 1. **Making music is easy, but standing out is hard**

* **Anyone can create a song**, but **millions of songs are uploaded every day**. Fame comes from being noticed, not just producing content.
* Musicians who break through often have **a unique voice, style, or story** that connects emotionally with an audience. Even technically perfect songs can go unnoticed if they don’t resonate.

---

### 2. **Distribution and exposure matter**

* Platforms like Spotify, YouTube, TikTok, and Instagram let anyone release music—but **attention is limited**.
* Going viral often depends on timing, trends, algorithmic favor, and luck, not just quality. Many excellent songs never reach wide audiences.
* Collaborations with other artists, influencers, or playlists can amplify reach.

---

### 3. **Building a personal brand**

* Fame isn’t just about songs—it’s about **identity**. Fans follow people, not just tracks.
* Musicians often become known for **their personality, image, or lifestyle**, which is shared across social media and media appearances.
* Think of Billie Eilish’s distinctive style or Bad Bunny’s social media presence—they’re as recognizable for their persona as their music.

---

### 4. **Monetization is multi-layered**

Even if music creation is easy, **making money requires multiple revenue streams**:

1. **Streaming royalties**: Usually small per stream. Millions of streams are needed to earn a decent living.
2. **Live performances**: Tours, concerts, and festivals can bring in the bulk of income.
3. **Merchandise**: Clothing, posters, limited edition items.
4. **Sponsorships and brand deals**: Popular musicians partner with brands for income.
5. **Songwriting/production credits**: Other artists or media using your music can generate royalties.
6. **Sync licensing**: Music in TV, movies, or ads pays well if your song gets picked.

---

### 5. **Networking and industry knowledge**

* Connections with producers, managers, record labels, and other artists still matter. Even with digital platforms, **industry support accelerates exposure**.
* Musicians who treat it like a business—understanding contracts, promotion, and analytics—tend to survive and thrive longer.

---

### 6. **Consistency and persistence**

* Fame is rarely instant. Even artists who “explode” often spent **years honing skills, building a fanbase, and releasing content**.
* Consistent output and engagement with fans keep an artist visible in a crowded market.

---

✅ **In short:**
Easy access to tools has lowered the barrier to entry, but **fame and income come from a combination of talent, uniqueness, marketing, networking, and sometimes luck**. Music creation is just one small piece of the puzzle.

---

If you want, I can make a **step-by-step blueprint showing how an independent musician today could realistically become famous and earn money**, including timelines and practical actions. That might make this much clearer.

Do you want me to do that?
""",
                """

Engineering work can actually span both the **services** and **goods** sides of the economy, depending on what exactly is being produced. Let me break it down clearly:

1. **Engineering as a service:**

   * When engineers provide **consulting, design, planning, or problem-solving** for clients, it’s considered a service.
   * Examples:

     * Designing a bridge or a building
     * Providing software engineering solutions
     * Engineering consulting for a manufacturing process
   * These are intangible outputs—you’re paying for expertise, knowledge, and intellectual labor rather than a physical product.

2. **Engineering as part of goods production:**

   * When engineers are directly involved in **creating physical products**, their work contributes to the goods economy.
   * Examples:

     * Designing and building machinery
     * Developing a new consumer electronic device
     * Engineering parts for cars, planes, or industrial equipment
   * Here, engineering is tied to the production of tangible goods.

✅ **Bottom line:** Engineering itself is often **a service**, because it’s the application of expertise, but it is also **closely linked to goods production** when that expertise results in a physical product. In economic classification, most engineering is counted under the **service sector**, even if it supports manufacturing.

If you want, I can make a **simple diagram showing engineering in services vs goods economy**—it’s easier to see visually. Do you want me to do that?
In basic biological classification, there are **five main classes of vertebrates**:

1. **Fish** – cold-blooded, live in water, usually have scales and fins
2. **Amphibians** – cold-blooded, live part of their life in water and part on land (e.g., frogs)
3. **Reptiles** – cold-blooded, mostly land-dwelling, have dry scaly skin (e.g., snakes, lizards)
4. **Birds (Aves)** – warm-blooded, have feathers and lay eggs
5. **Mammals** – warm-blooded, have hair or fur, and produce milk for their young

👉 **Note:** In more advanced taxonomy, fish are sometimes split into multiple classes (such as jawless, cartilaginous, and bony fish), which increases the total number. But at the school level, the answer is **five classes**.
Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"
Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"
Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"
Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"
Caroline: "Yeah, I'm really lucky to have them. They've been there through everything, I've known these friends for 4 years, since I moved from my home country. Their love and help have been so important especially after that tough breakup. I'm super thankful. Who supports you, Mel?"

Ah, this is a really interesting question because it’s deceptively tricky. On the surface, it might seem like “anyone can make music” these days—smartphones, DAWs (digital audio workstations), and AI-assisted tools have made production easier than ever. But becoming famous and making a sustainable income is still **much harder**. Let me break it down carefully:

---

### 1. **Making music is easy, but standing out is hard**

* **Anyone can create a song**, but **millions of songs are uploaded every day**. Fame comes from being noticed, not just producing content.
* Musicians who break through often have **a unique voice, style, or story** that connects emotionally with an audience. Even technically perfect songs can go unnoticed if they don’t resonate.

---

### 2. **Distribution and exposure matter**

* Platforms like Spotify, YouTube, TikTok, and Instagram let anyone release music—but **attention is limited**.
* Going viral often depends on timing, trends, algorithmic favor, and luck, not just quality. Many excellent songs never reach wide audiences.
* Collaborations with other artists, influencers, or playlists can amplify reach.

---

### 3. **Building a personal brand**

* Fame isn’t just about songs—it’s about **identity**. Fans follow people, not just tracks.
* Musicians often become known for **their personality, image, or lifestyle**, which is shared across social media and media appearances.
* Think of Billie Eilish’s distinctive style or Bad Bunny’s social media presence—they’re as recognizable for their persona as their music.

---

### 4. **Monetization is multi-layered**

Even if music creation is easy, **making money requires multiple revenue streams**:

1. **Streaming royalties**: Usually small per stream. Millions of streams are needed to earn a decent living.
2. **Live performances**: Tours, concerts, and festivals can bring in the bulk of income.
3. **Merchandise**: Clothing, posters, limited edition items.
4. **Sponsorships and brand deals**: Popular musicians partner with brands for income.
5. **Songwriting/production credits**: Other artists or media using your music can generate royalties.
6. **Sync licensing**: Music in TV, movies, or ads pays well if your song gets picked.

---

### 5. **Networking and industry knowledge**

* Connections with producers, managers, record labels, and other artists still matter. Even with digital platforms, **industry support accelerates exposure**.
* Musicians who treat it like a business—understanding contracts, promotion, and analytics—tend to survive and thrive longer.

---

### 6. **Consistency and persistence**

* Fame is rarely instant. Even artists who “explode” often spent **years honing skills, building a fanbase, and releasing content**.
* Consistent output and engagement with fans keep an artist visible in a crowded market.

---

✅ **In short:**
Easy access to tools has lowered the barrier to entry, but **fame and income come from a combination of talent, uniqueness, marketing, networking, and sometimes luck**. Music creation is just one small piece of the puzzle.

---

If you want, I can make a **step-by-step blueprint showing how an independent musician today could realistically become famous and earn money**, including timelines and practical actions. That might make this much clearer.

Do you want me to do that?
""",
                """

Engineering work can actually span both the **services** and **goods** sides of the economy, depending on what exactly is being produced. Let me break it down clearly:

1. **Engineering as a service:**

   * When engineers provide **consulting, design, planning, or problem-solving** for clients, it’s considered a service.
   * Examples:

     * Designing a bridge or a building
     * Providing software engineering solutions
     * Engineering consulting for a manufacturing process
   * These are intangible outputs—you’re paying for expertise, knowledge, and intellectual labor rather than a physical product.

2. **Engineering as part of goods production:**

   * When engineers are directly involved in **creating physical products**, their work contributes to the goods economy.
   * Examples:

     * Designing and building machinery
     * Developing a new consumer electronic device
     * Engineering parts for cars, planes, or industrial equipment
   * Here, engineering is tied to the production of tangible goods.

✅ **Bottom line:** Engineering itself is often **a service**, because it’s the application of expertise, but it is also **closely linked to goods production** when that expertise results in a physical product. In economic classification, most engineering is counted under the **service sector**, even if it supports manufacturing.

If you want, I can make a **simple diagram showing engineering in services vs goods economy**—it’s easier to see visually. Do you want me to do that?
In basic biological classification, there are **five main classes of vertebrates**:

1. **Fish** – cold-blooded, live in water, usually have scales and fins
2. **Amphibians** – cold-blooded, live part of their life in water and part on land (e.g., frogs)
3. **Reptiles** – cold-blooded, mostly land-dwelling, have dry scaly skin (e.g., snakes, lizards)
4. **Birds (Aves)** – warm-blooded, have feathers and lay eggs
5. **Mammals** – warm-blooded, have hair or fur, and produce milk for their young

👉 **Note:** In more advanced taxonomy, fish are sometimes split into multiple classes (such as jawless, cartilaginous, and bony fish), which increases the total number. But at the school level, the answer is **five classes**.

Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."
Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."
Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."
Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."
Caroline: "Thanks, Melanie! This necklace is super special to me - a gift from my grandma in my home country, Sweden. She gave it to me when I was young, and it stands for love, faith and strength. It's like a reminder of my roots and all the love and support I get from my family."

Ah, this is a really interesting question because it’s deceptively tricky. On the surface, it might seem like “anyone can make music” these days—smartphones, DAWs (digital audio workstations), and AI-assisted tools have made production easier than ever. But becoming famous and making a sustainable income is still **much harder**. Let me break it down carefully:

---

### 1. **Making music is easy, but standing out is hard**

* **Anyone can create a song**, but **millions of songs are uploaded every day**. Fame comes from being noticed, not just producing content.
* Musicians who break through often have **a unique voice, style, or story** that connects emotionally with an audience. Even technically perfect songs can go unnoticed if they don’t resonate.

---

### 2. **Distribution and exposure matter**

* Platforms like Spotify, YouTube, TikTok, and Instagram let anyone release music—but **attention is limited**.
* Going viral often depends on timing, trends, algorithmic favor, and luck, not just quality. Many excellent songs never reach wide audiences.
* Collaborations with other artists, influencers, or playlists can amplify reach.

---

### 3. **Building a personal brand**

* Fame isn’t just about songs—it’s about **identity**. Fans follow people, not just tracks.
* Musicians often become known for **their personality, image, or lifestyle**, which is shared across social media and media appearances.
* Think of Billie Eilish’s distinctive style or Bad Bunny’s social media presence—they’re as recognizable for their persona as their music.

---

### 4. **Monetization is multi-layered**

Even if music creation is easy, **making money requires multiple revenue streams**:

1. **Streaming royalties**: Usually small per stream. Millions of streams are needed to earn a decent living.
2. **Live performances**: Tours, concerts, and festivals can bring in the bulk of income.
3. **Merchandise**: Clothing, posters, limited edition items.
4. **Sponsorships and brand deals**: Popular musicians partner with brands for income.
5. **Songwriting/production credits**: Other artists or media using your music can generate royalties.
6. **Sync licensing**: Music in TV, movies, or ads pays well if your song gets picked.

---

### 5. **Networking and industry knowledge**

* Connections with producers, managers, record labels, and other artists still matter. Even with digital platforms, **industry support accelerates exposure**.
* Musicians who treat it like a business—understanding contracts, promotion, and analytics—tend to survive and thrive longer.

---

### 6. **Consistency and persistence**

* Fame is rarely instant. Even artists who “explode” often spent **years honing skills, building a fanbase, and releasing content**.
* Consistent output and engagement with fans keep an artist visible in a crowded market.

---

✅ **In short:**
Easy access to tools has lowered the barrier to entry, but **fame and income come from a combination of talent, uniqueness, marketing, networking, and sometimes luck**. Music creation is just one small piece of the puzzle.

---

If you want, I can make a **step-by-step blueprint showing how an independent musician today could realistically become famous and earn money**, including timelines and practical actions. That might make this much clearer.

Do you want me to do that?
""",
                """Caroline""",
            ],
        )
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
