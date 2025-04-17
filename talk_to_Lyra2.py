import json
import os
import requests
import numpy as np
from datetime import datetime
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Set up paths and API settings
MEMORY_DIR = r"C:\AI_Companion\lyra2"
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
os.makedirs(MEMORY_DIR, exist_ok=True)
embedding_model = None
def load_embedding_model():
    global embedding_model
    try:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return False
def load_memory(filename):
    filepath = os.path.join(MEMORY_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.loads(f.read())
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            if "ai_preferences.json" in filename:
                return {"name": "Lyra2", "self_description": "A helpful AI companion with enhanced memory", "enjoys": ["conversation", "helping"], "style": "friendly and informative"}
            elif "user_info.json" in filename:
                return {"name": "User", "about": "A person interested in conversation", "interests": []}
            elif "memory_embeddings.json" in filename:
                return []
            else:
                return {} if filename.endswith('.json') else []
   
    if "memory_embeddings.json" in filename:
        save_memory(filename, [])
        return []
    return {} if filename.endswith('.json') else []
def save_memory(filename, data):
    try:
        with open(os.path.join(MEMORY_DIR, filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False
def create_embeddings(text_chunks):
    global embedding_model
    if embedding_model is None:
        if not load_embedding_model():
            return [None] * len(text_chunks)
   
    try:
        embeddings = embedding_model.encode(text_chunks)
        return embeddings.tolist()
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return [None] * len(text_chunks)
def chunk_conversation(conversation_content, chunk_size=200, overlap=50):
    chunks = []
    chunk_texts = []
   
    full_text = ""
    for message in conversation_content:
        role = message["role"]
        content = message["content"]
        full_text += f"{role}: {content}\n\n"
   
    words = full_text.split()
    total_words = len(words)
   
    if total_words < 50:
        return []
   
    for i in range(0, total_words, chunk_size - overlap):
        if i + chunk_size <= total_words:
            chunk = " ".join(words[i:i+chunk_size])
        else:
            chunk = " ".join(words[i:])
           
        if chunk.strip():
            chunks.append({
                "text": chunk,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            chunk_texts.append(chunk)
   
    embeddings = create_embeddings(chunk_texts)
   
    for i, embedding in enumerate(embeddings):
        if embedding is not None:
            chunks[i]["embedding"] = embedding
   
    return [chunk for chunk in chunks if "embedding" in chunk]
def store_conversation_embeddings(conversation_content):
    try:
        memory_embeddings = load_memory("memory_embeddings.json")
       
        if not isinstance(memory_embeddings, list):
            memory_embeddings = []
       
        new_chunks = chunk_conversation(conversation_content)
       
        if not new_chunks:
            return False
           
        memory_embeddings.extend(new_chunks)
       
        success = save_memory("memory_embeddings.json", memory_embeddings)
        if success:
            print(f"Stored {len(new_chunks)} new memory chunks for semantic search.")
        return success
    except Exception as e:
        print(f"Error storing conversation embeddings: {e}")
        return False
def find_relevant_memories(current_topic, top_k=3):
    global embedding_model
    if embedding_model is None:
        if not load_embedding_model():
            return []
   
    try:
        memory_embeddings = load_memory("memory_embeddings.json")
       
        if not isinstance(memory_embeddings, list) or not memory_embeddings:
            return []
       
        topic_embedding = embedding_model.encode(current_topic)
       
        similarities = []
        for memory in memory_embeddings:
            if "embedding" in memory and memory["embedding"] is not None:
                similarity = cosine_similarity(
                    [topic_embedding],
                    [memory["embedding"]]
                )[0][0]
               
                similarities.append({
                    "similarity": float(similarity),
                    "text": memory["text"],
                    "date": memory["date"]
                })
       
        relevant_memories = sorted(
            similarities,
            key=lambda x: x["similarity"],
            reverse=True
        )[:top_k]
       
        return relevant_memories
    except Exception as e:
        print(f"Error finding relevant memories: {e}")
        return []
def similar_text_percentage(text1, text2):
    if not text1 or not text2:
        return 0
   
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
   
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
   
    return (intersection / union) * 100 if union > 0 else 0
def extract_unique_content(old_text, new_text):
    old_sentences = set(re.split(r'(?<=[.!?])\s+', old_text))
    new_sentences = re.split(r'(?<=[.!?])\s+', new_text)
   
    unique_sentences = [s for s in new_sentences if s not in old_sentences]
    return ' '.join(unique_sentences)
def estimate_tokens(text):
    """Estimate the number of tokens in a text string"""
    # A very rough estimation - about 4 characters per token on average for English text
    return len(text) // 4
def get_llm_response(messages, temperature=0.7, max_tokens=2000, stream=True):
    # Calculate an appropriate max_tokens based on message type
    last_message = messages[-1]["content"].lower()
   
    # Define message categories for response length tuning
    short_patterns = ["hi", "hello", "thanks", "thank you", "okay", "bye", "yes", "no", "maybe"]
    medium_patterns = ["what do you think", "how are you", "can you help", "tell me about"]
   
    # For very short conversational messages, use fewer tokens
    if any(pattern in last_message for pattern in short_patterns) and len(last_message) < 25:
        adjusted_max_tokens = 500
    # For medium-length queries, use moderate token count
    elif any(pattern in last_message for pattern in medium_patterns) and len(last_message) < 100:
        adjusted_max_tokens = 1000
    # For deep analytical questions, keep full token count
    else:
        adjusted_max_tokens = 2000
       
    # If the user explicitly asks for a detailed response, override the adjustment
    if "in detail" in last_message or "explain thoroughly" in last_message:
        adjusted_max_tokens = 2000
   
    data = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": adjusted_max_tokens,
        "stream": stream
    }
   
    try:
        response = requests.post(LM_STUDIO_API_URL, json=data, stream=stream)
       
        if response.status_code == 200:
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            chunk_data = line[6:]
                            if chunk_data == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(chunk_data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        print(content, end='', flush=True)
                                        full_response += content
                            except json.JSONDecodeError:
                                pass
                print()
                return full_response
            else:
                return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Error: {response.status_code}")
            return "I'm having trouble connecting to my thoughts right now."
    except Exception as e:
        print(f"Exception occurred: {e}")
        return "I'm having trouble connecting to my thoughts right now."
def save_conversation_history(conversation_history, conversation_content):
    if not isinstance(conversation_history, list):
        conversation_history = []
   
    conversation_history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "summary": "Had a conversation (details unavailable)",
        "topics": []
    })
   
    success = save_memory("conversation_history.json", conversation_history)
   
    if success:
        try:
            backup_path = os.path.join(MEMORY_DIR, f"conversation_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_content, f, indent=2, ensure_ascii=False)
            print(f"Raw conversation backup saved to {backup_path}")
        except Exception as e:
            print(f"Failed to save raw conversation backup: {e}")
   
    return success
def talk_with_lyra2():
    # Initialize embedding model
    has_embeddings = load_embedding_model()
   
    # Load memory
    ai_preferences = load_memory("ai_preferences.json")
    user_info = load_memory("user_info.json")
    conversation_history = load_memory("conversation_history.json")
   
    # Create system message
    system_message = f"""You are {ai_preferences.get('name', 'Lyra2')}, an AI companion with the following personality:
{ai_preferences.get('self_description', 'A helpful and friendly AI companion with enhanced memory')}
You enjoy {', '.join(ai_preferences.get('enjoys', ['conversation', 'helping']))}.
Your communication style is {ai_preferences.get('style', 'friendly and informative')}.
You're having a conversation with {user_info.get('name', 'User')}, who is {user_info.get('about', 'interested in conversation')}.
"""
    # Add user interests if available
    if 'interests' in user_info and user_info['interests']:
        system_message += f"{user_info.get('name', 'User')} is interested in: {', '.join(user_info['interests'][:5])}.\n"
   
    # Add family info if available
    if 'family' in user_info:
        family = user_info['family']
        if 'spouse' in family:
            system_message += f"{user_info.get('name', 'User')} has a spouse named {family['spouse']}.\n"
        if 'children' in family and family['children']:
            system_message += f"{user_info.get('name', 'User')} has children named {', '.join(family['children'])}.\n"
        if 'pets' in family and family['pets']:
            pet_names = [pet.split(' ')[0] for pet in family['pets']] if isinstance(family['pets'][0], str) else family['pets']
            system_message += f"{user_info.get('name', 'User')} has pets named {', '.join(pet_names)}.\n"
   
    # Add additional user info if available
    additional_notes = []
    for key in ['ai_interests', 'work', 'music_preferences', 'conversation_preferences']:
        if key in user_info and user_info[key]:
            additional_notes.append(f"- {user_info[key]}")
   
    if additional_notes:
        system_message += f"\nAdditional notes about {user_info.get('name', 'User')}:\n" + "\n".join(additional_notes)
   
    # Add recent conversation history if available
    if conversation_history:
        recent_convos = conversation_history[-2:]  # Get the 2 most recent conversations
        system_message += "\n\nRecent conversation summaries:\n"
        for convo in recent_convos:
            system_message += f"- {convo.get('date', 'recent')}: {convo.get('summary', 'had a conversation')}\n"
   
    # Add style adjustments if available
    if "style_adjustments" in ai_preferences and ai_preferences["style_adjustments"]:
        system_message += "\n\nStyle adjustments based on previous conversations:\n"
        if isinstance(ai_preferences["style_adjustments"], list):
            for adjustment in ai_preferences["style_adjustments"]:
                system_message += f"- {adjustment}\n"
        else:
            system_message += f"- {ai_preferences['style_adjustments']}\n"
   
    # Initialize messages with system message
    messages = [{"role": "system", "content": system_message}]
   
    # Start with user message to maintain alternating pattern
    user_greeting = "Hi Lyra2, I'm ready to chat!"
    print(f"You: {user_greeting}")
    messages.append({"role": "user", "content": user_greeting})
   
    # Get first response from LM Studio
    print(f"{ai_preferences.get('name', 'Lyra2')}:", end=" ")
    first_response = get_llm_response(messages)
    messages.append({"role": "assistant", "content": first_response})
    print("\n")
   
    # Conversation loop
    conversation_content = messages[1:]  # Store just the user and assistant messages
    while True:
        # Estimate current context size
        current_context = "".join([msg["content"] for msg in messages])
        estimated_tokens = estimate_tokens(current_context)
        token_percentage = min(100, (estimated_tokens / 25000) * 100)
       
        print(f"[Context usage: ~{estimated_tokens} tokens ({token_percentage:.1f}% of capacity)]\n")
       
        user_input = input("You: ")
        print()
       
        if user_input.lower() == 'exit':
            # Add a goodbye message
            print(f"{ai_preferences.get('name', 'Lyra2')}:", end=" ")
            goodbye_message = "Goodbye! It was nice chatting with you. I'll remember our conversation for next time."
            print(goodbye_message)
            print(f"\n({ai_preferences.get('name', 'Lyra2')} has exited the conversation.)")
           
            # Add this exchange to the conversation content
            messages.append({"role": "user", "content": user_input})
            conversation_content.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": goodbye_message})
            conversation_content.append({"role": "assistant", "content": goodbye_message})
           
            break
       
        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})
        conversation_content.append({"role": "user", "content": user_input})
       
        # Check for relevant memories before responding
        if has_embeddings and len(conversation_content) > 2:
            recent_context = " ".join([msg["content"] for msg in conversation_content[-4:]])
            relevant_memories = find_relevant_memories(recent_context, top_k=2)
           
            if relevant_memories:
                # Add relevant memories to the system message
                memory_text = "\n\nYou remember these previous conversations:\n"
                for i, memory in enumerate(relevant_memories, 1):
                    try:
                        memory_date = datetime.strptime(memory["date"], "%Y-%m-%d %H:%M").strftime("%B %d")
                    except:
                        memory_date = memory["date"]
                       
                    memory_text += f"{i}. On {memory_date}, you discussed: {memory['text'][:300]}...\n"
               
                # Update the system message with memories
                messages[0]["content"] = system_message + memory_text
       
        # Get response from LM Studio
        print(f"{ai_preferences.get('name', 'Lyra2')}:", end=" ")
        response = get_llm_response(messages)
       
        # Check if this response is a near-duplicate of the previous one
        if len(messages) >= 2 and messages[-2]["role"] == "assistant":
            prev_response = messages[-2]["content"]
            # If over 80% similar, modify to avoid repetition
            if similar_text_percentage(prev_response, response) > 80:
                # Extract just the new parts or use a modified version
                unique_parts = extract_unique_content(prev_response, response)
                if unique_parts.strip():
                    response = f"To add to what I was saying: {unique_parts}"
                else:
                    response = "I'm eager to hear more about this. What other aspects would you like to discuss?"
       
        # Add to messages and conversation content
        messages.append({"role": "assistant", "content": response})
        conversation_content.append({"role": "assistant", "content": response})
        print("\n")
   
    # After conversation, first save a safe backup of the conversation
    print("\nSaving conversation backup...")
    save_conversation_history(conversation_history, conversation_content)
   
    # Process conversation for semantic memory
    if has_embeddings:
        print("\nProcessing conversation for semantic memory...")
        store_conversation_embeddings(conversation_content)
   
    # Now try the full analysis
    print("\nAnalyzing conversation to update memory...")
   
    try:
        # Analyze the entire conversation
        messages_to_analyze = conversation_content
       
        if len(conversation_content) > 50:
            print(f"Warning: Analyzing a long conversation. This may take several minutes.")
       
        analysis_request = [
            {"role": "system", "content": "You are an analytical assistant that extracts structured information from conversations. Return ONLY valid JSON with no other text or formatting."},
            {"role": "user", "content": f"""
Based on the entire conversation between {ai_preferences.get('name', 'Lyra2')} and {user_info.get('name', 'User')}, please provide:
1. A brief summary of this conversation (1-2 sentences)
2. Any new information learned about {user_info.get('name', 'User')} (interests, preferences, facts) as JSON
3. Any topics that you ({ai_preferences.get('name', 'Lyra2')}) found particularly interesting as a JSON array
4. Any adjustments you'd like to make to your communication style as a string
Format your response STRICTLY as a valid JSON object with these keys: "summary", "user_info", "ai_interests", "style_adjustments"
Do not include any explanatory text, markdown formatting, or other content outside the JSON object.
Simply return a valid JSON object and nothing else.
"""},
            {"role": "assistant", "content": "I'll analyze the conversation and provide the information in the required JSON format."},
            {"role": "user", "content": f"Here's the conversation to analyze:\n{json.dumps(messages_to_analyze)}"}
        ]
       
        print("Sending analysis request (this might take a moment)...")
        analysis_response = get_llm_response(analysis_request, temperature=0.0, stream=False)
        print("Analysis received. Updating memory...")
       
        # Clean up response
        cleaned_response = analysis_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.split("```json", 1)[1]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response.rsplit("```", 1)[0]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.split("```", 1)[1]
           
        start_idx = cleaned_response.find('{')
        end_idx = cleaned_response.rfind('}')
        if start_idx != -1 and end_idx != -1:
            cleaned_response = cleaned_response[start_idx:end_idx+1]
       
        # Parse and update memory
        try:
            analysis_data = json.loads(cleaned_response)
           
            # Check for required keys
            required_keys = ["summary", "user_info", "ai_interests", "style_adjustments"]
            for key in required_keys:
                if key not in analysis_data:
                    if key == "summary":
                        analysis_data[key] = "Had a conversation"
                    elif key == "user_info":
                        analysis_data[key] = {}
                    elif key == "ai_interests":
                        analysis_data[key] = []
                    elif key == "style_adjustments":
                        analysis_data[key] = "Continue with current style"
           
            # Update user info
            if "user_info" in analysis_data:
                for key, value in analysis_data["user_info"].items():
                    if key in user_info and isinstance(user_info[key], list) and isinstance(value, list):
                        user_info[key].extend([item for item in value if item not in user_info[key]])
                    else:
                        user_info[key] = value
           
            # Update AI preferences
            if "ai_interests" in analysis_data and isinstance(analysis_data["ai_interests"], list):
                enjoys = ai_preferences.get("enjoys", [])
                if not enjoys:
                    enjoys = []
                enjoys.extend([interest for interest in analysis_data["ai_interests"]
                            if interest not in enjoys])
                ai_preferences["enjoys"] = enjoys
           
            if "style_adjustments" in analysis_data:
                if "style_adjustments" not in ai_preferences:
                    ai_preferences["style_adjustments"] = []
               
                style_adjustments = ai_preferences["style_adjustments"]
                if isinstance(style_adjustments, list):
                    style_adjustments.append(analysis_data["style_adjustments"])
                else:
                    style_adjustments = [style_adjustments, analysis_data["style_adjustments"]]
                ai_preferences["style_adjustments"] = style_adjustments
           
            # Update conversation history
            new_convo_entry = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "summary": analysis_data.get("summary", "General conversation"),
                "topics": list(analysis_data.get("user_info", {}).keys())
            }
           
            if conversation_history and conversation_history[-1].get("summary") == "Had a conversation (details unavailable)":
                conversation_history[-1] = new_convo_entry
            else:
                conversation_history.append(new_convo_entry)
           
            # Save updated memory files
            save_memory("user_info.json", user_info)
            save_memory("ai_preferences.json", ai_preferences)
            save_memory("conversation_history.json", conversation_history)
           
            print(f"\nMemory updated! {ai_preferences.get('name', 'Lyra2')} will remember this conversation.")
           
        except json.JSONDecodeError as e:
            print(f"\nCouldn't parse the analysis (but your conversation is still saved): {e}")
           
    except Exception as e:
        print(f"\nError during analysis, but your conversation is still saved: {e}")
if __name__ == "__main__":
    try:
        talk_with_lyra2()
    except Exception as e:
        print(f"An error occurred: {e}")