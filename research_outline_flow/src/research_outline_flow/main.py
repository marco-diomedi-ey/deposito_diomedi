#!/usr/bin/env python
import json
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

from research_outline_flow.crews.outline_crew.outline_crew import OutlineCrew
from research_outline_flow.crews.research_crew.research_crew import ResearchCrew
from research_outline_flow.crews.ethical_crew.ethical_crew import EthicalCrew
from research_outline_flow.crews.analyst_crew.analyst_crew import AnalystCrew
from research_outline_flow.crews.math_crew.math_crew import MathCrew


def extract_json_from_text(text):
    # Trovare la prima [ e l'ultima ] che chiude l'array
    start = text.find('[')
    if start == -1:
        return text  # Fallback se non è un array
    
    bracket_count = 0
    end = start
    
    for i in range(start, len(text)):
        if text[i] == '[':
            bracket_count += 1
        elif text[i] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end = i + 1
                break
    
    return text[start:end]



class OutlineState(BaseModel):
    question: str = ""
    bullet_points: list[str] = []
    is_ethical: bool = False
    ethical_analysis: str = ""
    bias_analysis: str = ""
    has_bias: bool = False
    reviewed_outline: list[str] = []
    paragraphs: list[str] = []
    outline: str = ""
    answer: str = ""
    is_math_related: bool = False


class OutlineFlow(Flow[OutlineState]):

    @start("retry")
    def start(self):
        print("Starting the Answer Flow")
        return "starting"

    @listen(start)
    def generate_question(self):
        print("Generating question")
        self.state.question = input("Enter your question: ")
    
    @router(generate_question)
    def generate_analysis(self):
        print("Generating analysis")
        result = (
            AnalystCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
            })
        )
        cleaned_raw = result.raw.strip()
        if cleaned_raw.lower() == "true":
            return "is_math_related"
        elif cleaned_raw.lower() == "false":
            return "is_not_math_related"
        else:
            print(f"Warning: unexpected output from analysis task: {cleaned_raw}")
            self.state.is_math_related = False  # Default fallback
        print(f"Is the question math related? {self.state.is_math_related}")
    
    @listen("is_math_related")
    def generate_math_outline(self):
        print("Generating math outline")
        result = (
            MathCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
                "outline": "",  # Valore vuoto per task che non lo usano
                "bullet_point": ""  # Valore vuoto per task che non lo usano
            })
        )
        print("Math outline generated:")
    

    @listen(generate_math_outline)
    def execute_math_problem(self):
        print("Executing math problem")
        result = (
            MathCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
                "outline": "",  # Valore vuoto per task che non lo usano
                "bullet_point": ""  # Valore vuoto per task che non lo usano
            })
        )
        print("Math problem executed successfully.")

    @listen("is_not_math_related")
    def generate_ethical_analysis(self):
        print("Generating ethical analysis")
        result = (
            EthicalCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
            })
        )
        cleaned_raw = result.raw.strip()
        if cleaned_raw.startswith("```json"):
            cleaned_raw = cleaned_raw[7:]  # Rimuovi ```json
        if cleaned_raw.endswith("```"):
            cleaned_raw = cleaned_raw[:-3]  # Rimuovi ```
        cleaned_raw = cleaned_raw.strip()
        self.answer_json = json.loads(cleaned_raw)
        self.state.ethical_analysis = self.answer_json["reasoning"]
        self.state.is_ethical = self.answer_json["is_ethical"]

    @router(generate_ethical_analysis)
    def route_ethical_analysis(self):
        print("Routing ethical analysis")
        if self.state.is_ethical:
            return "ethical"
        else:
            print("The question is not ethical, I cannot answer it.")
            return "retry"

    @listen("ethical")
    def generate_outline(self):
        print("Generating outline")
        result = (
            OutlineCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
                "outline": "",  # Valore vuoto per task che non lo usano
                "bullet_point": ""  # Valore vuoto per task che non lo usano
            })
        )
        self.state.bullet_points = [result.raw]
    
    @listen(generate_outline)
    def review_outline(self):
        print("Reviewing outline")
        print("Here is the generated outline:")
        for i, point in enumerate(self.state.bullet_points):
            print(f"{i+1}. {point}")
        result = (
            OutlineCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
                "outline": self.state.bullet_points,
                "bullet_point": ""  # Valore vuoto per task che non lo usano
            })
        )
        # Nel codice:
        cleaned_raw = result.raw.strip()
        if cleaned_raw.startswith("```json"):
            cleaned_raw = cleaned_raw[7:]  # Rimuovi ```json
        if cleaned_raw.endswith("```"):
            cleaned_raw = cleaned_raw[:-3]  # Rimuovi ```
        cleaned_raw = cleaned_raw.strip()
        cleaned_raw = extract_json_from_text(cleaned_raw)  # Rimuove testo extra
        self.answer_bias_json = json.loads(cleaned_raw)
        print(f"DEBUG - cleaned content: {self.answer_bias_json}")
        print(f"DEBUG - first 20 chars: {self.answer_bias_json[:20]}")
        print(f"DEBUG - last 20 chars: {self.answer_bias_json[-20:]}")
        self.state.reviewed_outline = [self.answer_bias_json]
        # self.answer_bias_json = json.loads(cleaned_raw)
        # Gestire il fatto che è un array:
        if isinstance(self.answer_bias_json, list) and len(self.answer_bias_json) > 0:
            # Aggregare tutti i bias results
            bias_summaries = []
            has_any_bias = False
            
            for item in self.answer_bias_json:
                bias_summaries.append(f"{item['name']}: {item['reasoning_bias']}")
                if item.get('has_bias', False):
                    has_any_bias = True
            
            self.state.bias_analysis = " | ".join(bias_summaries)
            self.state.has_bias = has_any_bias
            self.state.reviewed_outline = [item['name'] for item in self.answer_bias_json]
        else:
            # Fallback se non è un array o è vuoto
            self.state.bias_analysis = "No bias analysis available"
            # self.state.has_bias = False
            # self.state.reviewed_outline = []
        # self.state.bias_analysis = self.answer_bias_json["reasoning_bias"]
        # self.state.has_bias = self.answer_bias_json["has_bias"]
        # # self.state.reviewed_outline = [result.raw]
        # self.state.reviewed_outline = [self.answer_bias_json]
        print(f"Reviewed outline: {self.state.reviewed_outline}")
    
    @listen(review_outline)
    def generate_research(self):
         # Filtrare solo i bullet points che NON hanno bias
        names_without_bias = [
            item['name'] for item in self.answer_bias_json 
            if not item.get('has_bias', False)
        ]
        self.state.bullet_points = names_without_bias
        # for name in names_without_bias:
        #     print("Generating web research")
        result = (
            ResearchCrew()
            .crew()
            .kickoff(inputs={
                "question": self.state.question,
                "bullet_point": self.state.bullet_points,
                "outline": "",  # Valore vuoto per task che non lo usano,
                "paragraphs": []  # Valore vuoto per task che non lo usano
            })
        )
        self.state.paragraphs.append(result.raw)
        
    
    @listen(generate_research)
    def generate_document(self):
        print("Generating final document")
        result = (
            ResearchCrew()
            .crew()
            .kickoff(inputs={
                "paragraphs": self.state.paragraphs,
                "question": self.state.question,
                "bullet_point": self.state.bullet_points,
                "outline": ""
            })
        )
        self.state.answer = result.raw

def kickoff():
    outline_flow = OutlineFlow()
    outline_flow.kickoff()


def plot():
    outline_flow = OutlineFlow()
    outline_flow.plot()


if __name__ == "__main__":
    kickoff()
