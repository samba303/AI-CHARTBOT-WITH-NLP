import spacy
from nltk.chat.util import Chat, reflections

# Load spaCy's small English model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Define pairs of patterns and responses for NLTK Chat
pairs = [
    (r"hi|hello|hey", ["Hello! How can I assist you today?", "Hi there! What can I do for you?"]),
    (r"what is your name?", ["I am your friendly chatbot!", "You can call me Chatbot."]),
    (r"how are you?", ["I'm just a program, but I'm functioning perfectly! How about you?"]),
    (r"tell me about (.*)", ["Sure! Here's some information about %1.", "I can help you learn more about %1."]),
    (r"quit", ["Goodbye! Have a great day!", "See you later! Take care!"]),
]

# Initialize NLTK Chat
chatbot = Chat(pairs, reflections)


def process_user_input(user_input):
    """
    Process user input using spaCy for advanced NLP tasks like Named Entity Recognition (NER).
    """
    doc = nlp(user_input)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def main():
    print("Chatbot: Hi! I am a chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip().lower()
        if user_input == "quit":
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Process user input for NER and intent recognition
        entities = process_user_input(user_input)
        if entities:
            print(f"Chatbot: I noticed you mentioned these entities: {entities}")

        # Use NLTK Chat for predefined responses
        response = chatbot.respond(user_input)
        if response:
            print(f"Chatbot: {response}")
        else:
            print("Chatbot: I'm not sure how to respond to that. Can you rephrase?")


if __name__ == "__main__":
    main()