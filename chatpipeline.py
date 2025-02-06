'''

Chat pipeline that stores User & AI conversations serving as memory for LLM

'''

from transformers import pipeline


class ChatPipeline:
    def __init__(self, model, tokenizer, max_history=10):
        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        self.chat_history = []  # Stores past messages
        self.max_history = max_history  # Number of past messages to retain . try changing this if the LLM throws token related error
        self.context = None

    def clear_chat(self):
        self.chat_history=[]

    def add_context(self,context):
        self.context=context
        self.chat_history.append(f"User: context={self.context}")
        self.chat_history.append(f"AI: Context added to memory")

    def chat(self, user_input):
        """Generates a response while maintaining chat history."""

        if self.context in user_input:
            user_input=user_input.strip(self.context)

        # Append the new message to history
        self.chat_history.append(f"User: {user_input}")

        # Trim history if it exceeds max_history
        if len(self.chat_history) > self.max_history:
            #pass
            self.chat_history.pop(1)

        # Format the conversation history as input
        conversation = "\n".join(self.chat_history) + "\nAI:"

        # Generate response
        response = self.pipeline(conversation, max_length=2048, pad_token_id=50256,truncation=True)[0]["generated_text"]

        #print('response:',response,'\n')

        # Extract only the model's reply
        ai_reply = response.split("AI:")[-1].strip()

        # Store model's response in history
        self.chat_history.append(f"AI: {ai_reply}")

        with open('latest_chat_history.txt','w') as f:
            f.write("\n".join(self.chat_history))

        return ai_reply
