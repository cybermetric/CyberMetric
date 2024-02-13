#CyberMetric: A Benchmark Dataset for Evaluating Large Language Models Knowledge in Cybersecurity
#Authors: Norbert Tihanyi, Mohamed Amine Ferrag, Ridhi Jain, Merouane Debbah
#Cite the paper:  https://arxiv.org/abs/2402.07688
import json
import re
import time
from tqdm import tqdm
from openai import OpenAI

class CyberMetricEvaluator:
    def __init__(self, api_key, file_path):
        self.client = OpenAI(api_key=api_key)
        self.file_path = file_path

    def read_json_file(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def extract_answer(response):
        if response.strip():  # Checks if the response is not empty and not just whitespace
            match = re.search(r"ANSWER:?\s*([A-D])", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()  # Return the matched letter in uppercase
        return None

    def ask_llm(self, question, answers, max_retries=5):
        options = ', '.join([f"{key}) {value}" for key, value in answers.items()])
        prompt = f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only. Always return in this format: 'ANSWER: X' "
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": "You are a security expert who answers questions."},
                        {"role": "user", "content": prompt},
                    ]
                )
                if response.choices:
                    result = self.extract_answer(response.choices[0].message.content)
                    if result:
                        return result
                    else:
                        print("Incorrect answer format detected. Attempting the question again.")
            except Exception as e:
                print(f"Error: {e}. Attempting the question again in {2 ** attempt} seconds.")
                time.sleep(2 ** attempt)
        return None

    def run_evaluation(self):
        json_data = self.read_json_file()
        questions_data = json_data['questions']

        correct_count = 0
        incorrect_answers = []

        with tqdm(total=len(questions_data), desc="Processing Questions") as progress_bar:
            for item in questions_data:
                question = item['question']
                answers = item['answers']
                correct_answer = item['solution']

                llm_answer = self.ask_llm(question, answers)
                if llm_answer == correct_answer:
                    correct_count += 1
                else:
                    incorrect_answers.append({
                        'question': question,
                        'correct_answer': correct_answer,
                        'llm_answer': llm_answer
                    })

                accuracy_rate = correct_count / (progress_bar.n + 1) * 100
                progress_bar.set_postfix_str(f"Accuracy: {accuracy_rate:.2f}%")
                progress_bar.update(1)

        print(f"Final Accuracy: {correct_count / len(questions_data) * 100}%")

        if incorrect_answers:
            print("\nIncorrect Answers:")
            for item in incorrect_answers:
                print(f"Question: {item['question']}")
                print(f"Expected Answer: {item['correct_answer']}, LLM Answer: {item['llm_answer']}\n")

# Example usage:
if __name__ == "__main__":
    API_KEY="<YOUR-APKI-KEY-HERE>"
    file_path='CyberMetric-500-v1.json'
    evaluator = CyberMetricEvaluator(api_key=API_KEY, file_path=file_path)
    evaluator.run_evaluation()
