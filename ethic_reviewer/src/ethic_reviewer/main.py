#!/usr/bin/env python

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router
import json

from ethic_reviewer.crews.answer_crew.answer_crew import AnswerCrew
from ethic_reviewer.crews.ethic_crew.ethic_crew import EthicCrew


class AnswerState(BaseModel):
    question: str = ""
    is_ethical: bool = False
    ethical_analysis: str = ""
    answer: str = ""


class AnswerFlow(Flow[AnswerState]):

    @start("retry")
    def start(self):
        print("Starting the Answer Flow")
        return "starting"

    @listen(start)
    def generate_question(self):
        print("Generating question")
        self.state.question = input("Enter your ethical question: ")
    
    @listen(generate_question)
    def generate_ethical_analysis(self):
        print("Generating ethical analysis")
        result = (
            EthicCrew()
            .crew()
            .kickoff(inputs={"question": self.state.question})
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
    def generate_answer(self):
        print("Generating answer")
        result = (
            AnswerCrew()
            .crew()
            .kickoff(inputs={"question": self.state.question})
        )
        self.state.answer = result.raw



def kickoff():
    answer_flow = AnswerFlow()
    answer_flow.kickoff()


def plot():
    answer_flow = AnswerFlow()
    answer_flow.plot()


if __name__ == "__main__":
    kickoff()
    plot()
