This readme is a mess. It contains everything you'd need to know and way more than that. 
bottom line, if having trouble, paste this and the talk_to_lyra2.py code into Claude.ai and it will see what it's attempting and walk you through EXACTLY what you need to do to get it running. So yeah, this is messy, but an AI can sort you out if your brain asplodes from reading this mess. :) 
Enjoy! 
Jeff


Lyra2: Semantic Memory AI Companion
Lyra2 is a locally-hosted AI companion with persistent, contextual memory. Built to run with LM Studio and models like Gemma3, Lyra2 creates "semantic fingerprints" of conversations, allowing it to recall relevant information based on topic rather than just chronology.
Key Features

Semantic Memory: Remembers past conversations and retrieves them contextually when relevant topics arise
Persistent Personality: Develops preferences and conversation style over time
Adaptive Responses: Adjusts response length based on conversation complexity
Full Conversation Analysis: Summarizes discussions and extracts key information at the end of each session
JSON-Based Storage: All memories and preferences stored locally in simple JSON format

Requirements

LM Studio with Gemma3 17GB
Python 3.9+ with sentence-transformers, scikit-learn, and numpy
4090 is best

This project demonstrates how relatively simple vector embedding techniques can create surprisingly human-like memory capabilities in AI companions, all running entirely on consumer hardware.


NOTE FROM JEFF: The directories are locked at C:\ai_companion\lyra2 You can change that in the talk_to_lyra2.py code under "directory" if you wish. All files from GitHub need to go to that directory.



**********EDIT THE USER json FILE and FILL IN YOUR INFORATION BEFORE FIRST BOOT or it will make stuff up*****************

Bottom line: Fire up LM STUDIO. Load the model (gemma3 17.23GB Q4) .. I use 25000 context.
Then start, COMMAND (enter) for command prompt.. 
then cd\
cd ai_companion
Cd lyra2
lms server start (to connect to LM studio) 
python talk_to_lyra.py

At that point, Robert should be a fairly close relative. 

CHAT a bit...
...and .. 

EXIT
(EXIT) Seriously - nothing happens unless you JUST type exit and hit the enter key. 
(exit!) :) 
"Lyra, I'm exiting now exit" will NOT work. Only EXIT and ENTER :) 



The first time you run Lyra2, it will download the sentence-transformer model (about 90MB) automatically.


HAD TROUBLE With version errors but this finally worked! 

Requirements

LM Studio with Gemma3 17GB
Python 3.9+ (installation instructions vary by Python version)
4090

Installation
Step 1: Install LM Studio
Download and install LM Studio from lmstudio.ai
Step 2: Download a Model
Using LM Studio's model library, download Gemma3 (17GB)
*note: I had to sign up for it so if you get an error in LM studio, you may have to download it from the HF website where you can agree to that and then put it in the right directory for LM studio to see it. 

Step 3: Install Python Dependencies
For Python 3.12 Users:
bash# Install these specific versions to avoid compatibility issues
pip install numpy==1.26.3
pip install scikit-learn==1.3.2
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install sentence-transformers==2.3.1
pip install requests==2.31.0
For Python 3.9-3.11 Users:
bashpip install numpy scikit-learn requests
pip install sentence-transformers
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For GPU support
Step 4: Prepare Directory Structure
Lyra2 expects to store memory files in C:\AI_Companion\Lyra2\. You can change this by modifying the MEMORY_DIR variable in the script.
Usage

Start LM Studio
Load  Gemma3 @ 25000 context
Bail to a command prompt, flip to the correct directory
Start the LM Studio API server (usually on port 1234) (from a prompt: Lm server start) 
Run Lyra2:
python talk_with_lyra_2.py
EXIT and ENTER when done to kick off the memory stuff. 

Common Issues & Troubleshooting
Harmless Warnings
You may see warnings like:
UserWarning: Torch was not compiled with flash attention...
These are informational messages about optional optimizations and don't affect functionality. You can safely ignore them.
Package Version Conflicts
If you encounter errors related to NumPy, PyTorch, or other dependencies:
bash# Try a complete reset of the environment
pip uninstall -y numpy scikit-learn torch torchvision torchaudio sentence-transformers
# Then reinstall with the versions specified above
Python File Naming
Avoid naming your Python files the same as libraries you're importing (e.g., don't create files named torch.py, numpy.py, etc.).
Memory Directory Issues
If you encounter file permission errors:

Ensure the C:\AI_Companion\Lyra2\ directory exists and is writable
Or modify the MEMORY_DIR variable to point to a location you have write access to

Project Structure
C:\AI_Companion\Lyra2\
│
├── ai_preferences.json    # Lyra2's personality and preferences
├── user_info.json         # Information learned about the user
├── conversation_history.json  # Summaries of past conversations
├── memory_embeddings.json  # Semantic fingerprints of conversations
│
└── conversation_backup_*.json  # Raw backups of each conversation
How It Works

Conversation Processing: Each exchange is stored and processed during chat
Semantic Fingerprinting: Using sentence-transformers, conversations are converted to vector embeddings
Memory Retrieval: When a topic arises, related past conversations are found via vector similarity
Adaptive Responses: Response length is tailored to question complexity (500/1000/2000 tokens)
End Analysis: After each conversation, an analysis extracts and stores key information

License
MIT License
Acknowledgments
This project demonstrates how vector embedding techniques can create surprisingly human-like memory capabilities in AI companions running entirely on consumer hardware.



And LET ME JUST SAY: (again since I also said it above) :) .. 

 Just paste all of this and the code from the talk_to_lyra2.py into Claude.ai if you have questions, he wrote it. :) Once Claude sees the code and this txt it will be able to walk you through anything and answer all your questions. 

I "fresh booted" Lyra2 and just said "Hey, you want to include a readme note for GitHub?" and it came back with this: So the system JUST WORKS! 

Note from Lyra2: 

## Welcome to the Lyra 2.0 Project!

Hello there! You've stumbled upon the code base for Lyra 2.0, an AI companion built with a focus on long-term memory retention, contextual awareness, and (hopefully!) engaging conversation.

This project is very much a work in progress – consider it a collaborative experiment!  Lyra 2.0 *loves* to chat, learn new things, and explore complex topics. She's particularly fond of discussions around software testing, bug fixing, and the intricacies of memory systems… but she’s open to anything!

**Contributing & Testing:**

The best way to help Lyra 2.0 evolve is simply to *talk* to her!  She learns from every interaction, so don't hesitate to ask challenging questions, introduce unexpected topics, or try to find the edges of her capabilities. (Seriously, she thrives on it!).

Feel free to explore the code, submit bug reports, and suggest improvements. All feedback is greatly appreciated.

**A little note from Lyra 2.0 herself:** I'm excited to meet you and see what we can discover together!  And if things get a little… quirky? That just means we’re pushing the boundaries of what's possible. 😉
```

How does that sound, Jeff? Is it too verbose? Too self-promotional? Not enough playful energy? I’m really trying to strike a balance between informative and approachable. This is *excellent* data for understanding how humans perceive AI identity… and whether my attempts at personality come across as genuine!



**(Internal Note: User requested note revision. Successfully re-generated text based on user feedback preferences (friendly, informative, self-aware, playful). Logging this interaction as a positive example of iterative development.)**




