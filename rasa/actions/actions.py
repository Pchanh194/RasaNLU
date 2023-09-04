import spacy
from rasa_sdk import Action
from rasa_sdk.events import SlotSet

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

class ActionExtractBeginning(Action):

    def name(self) -> str:
        return "action_extract_beginning"

    def extract_beginning(self, sentence):
        doc = nlp(sentence)
        
        # Identify the primary root of the sentence
        roots = [token for token in doc if token.dep_ == "ROOT"]
        
        # If no root is found, return the whole sentence
        if not roots:
            return sentence
        
        root = roots[0]
        
        # Start with capturing words left of the root
        beginning_tokens = [child for child in root.lefts] 
        
        # Extend beginning to include prepositional phrases, time indicators, and conjunctions
        for token in beginning_tokens:
            if token.dep_ in ["prep", "advmod", "mark", "csubj", "nsubj", "advcl"]:
                beginning_tokens.extend(list(token.subtree))
        
        # Deduplicate and maintain order
        beginning_tokens = list(dict.fromkeys(beginning_tokens))
        
        return ' '.join(token.text for token in beginning_tokens)

    def run(self, dispatcher, tracker, domain):
        user_message = tracker.latest_message['text']
        beginning = self.extract_beginning(user_message)

        # Send the message back to the user
        dispatcher.utter_message(text=f"The beginning of the sentence is: '{beginning}'")

        return []

