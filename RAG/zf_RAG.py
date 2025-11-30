from openai import OpenAI
import dotenv 
import json

dotenv.load_dotenv() 


def generate_context_relevance_score(question, context):
    prompt = f"On a scale of 0 to 1, how relevant is the following context to the question?\n\nQuestion: {question}\n\nContext: {context}\n\nRelevance Score:"
    client = OpenAI()

    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rates the relevance of context to a question."},
            {"role": "user", "content": prompt}
        ],
    )
    
    score_text = response.choices[0].message.content.strip() 
    
    try:
        score = float(score_text)
        if 0 <= score <= 1: 
            return score
        else:
            raise ValueError("Score out of range")
    except ValueError:
        raise ValueError(f"Invalid score received: {score_text}")


def generate_answer_relevance_score(question, answer):
    prompt = f"""
    On a scale of 0 to 1, how relevant is the following context to the question?\n\n
    RETURNS only the Score as a float between 0 and 1.
    Question: {question}\n\n
    Context: {answer}\n\n
    Relevance Score:
    """
    client = OpenAI()

    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rates the relevance of context to a question."},
            {"role": "user", "content": prompt}
        ],
    )
    
    score_text = response.choices[0].message.content.strip() 
    
    try:
        score = float(score_text)
        if 0 <= score <= 1:
            return score
        else:
            raise ValueError("Score out of range")
    except ValueError:
        raise ValueError(f"Invalid score received: {score_text}")
    

def generate_faithfulness_score(context, answer):
    prompt = f"""
    On a scale of 0 to 1, how faithful is the following answer to the context?\n\nContext: {context}\n\nAnswer: {answer}\n\n
    Returns only the Score as a float between 0 and 1. 
    Faithfulness Score:
    """ 
    client = OpenAI()

    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rates the relevance of context to a question."},
            {"role": "user", "content": prompt}
        ],
    )
    
    score_text = response.choices[0].message.content.strip() 
    
    try:
        score = float(score_text)
        if 0 <= score <= 1:
            return score
        else:
            raise ValueError("Score out of range")
    except ValueError:
        raise ValueError(f"Invalid score received: {score_text}")

# load the json file 
original_query = "Calculate the Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years, showing the working."
answer = """
The Operating Efficiency Ratio (Opex ÷ Operating Income) for the last 3 fiscal years is:  
- 2021: 17.17% (13,510 ÷ 78,714)  
- 2022: 21.00% (15,724 ÷ 74,842)  
- 2023: 19.48% (16,425 ÷ 84,293)
"""

logs = ["9"] #! because this query actually created 4 logs
context_rel = [] 
for log in logs: 

    with open(f'00-data/logs/sections/test_{log}.json', 'r') as f:
        data = json.load(f) 
        question = data['expanded_query']
        context = "" 
        for section in data.get('results', []):
            print (f"records in section: {len(section.get('ranking', []))}")
            for record in section.get('ranking', []):
                context += record.get('text', '') + "\n"

    print ("Question:", question) 
    print ("Context:", context[:500])  # Print first 500 characters of context
    print ("=====================")
    print ("Context:", context[-500:])  # Print last 500 characters of context


    score = generate_context_relevance_score(question, context) 
    context_rel.append(score) 
    print ("Relevance Score:", score) 

print ("Context Relevance Scores:", context_rel) 
print ("Average Context Relevance Score:", sum(context_rel)/len(context_rel)) 


faithfulness = generate_faithfulness_score(context, answer) 
print ("Faithfulness Score:", faithfulness) 

answer_rel = generate_answer_relevance_score(original_query, answer) 
print ("Answer Relevance Score:", answer_rel) 