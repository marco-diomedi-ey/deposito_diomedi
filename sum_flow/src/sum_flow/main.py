#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from sum_flow.crews.poem_crew.poem_crew import PoemCrew


class PoemState(BaseModel):
    a: int = 0
    b: int = 0
    flag1: bool = False
    flag2: bool = False
    num_result: int = 0


class PoemFlow(Flow[PoemState]):

    @start()
    def ask_two_numbers(self):
        self.state.flag1 = True
        self.state.flag2 = True  # Aggiungi questa linea
        while self.state.flag1:
            print("Asking for the first number")
            a = input("Enter the first number: ")
            try:
                int(a)
                self.state.a = int(a)
                self.state.flag1 = False
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        while self.state.flag2:    
            print("Asking for the second number")
            b = input("Enter the second number: ")
            try:
                int(b)
                self.state.b = int(b)
                self.state.flag2 = False
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        print(f"Numbers received: a={self.state.a}, b={self.state.b}")

    @listen(ask_two_numbers)
    def execute_sum(self):
        print("executing sum")
        result = (
            PoemCrew()
            .crew()
            .kickoff(inputs={"a": self.state.a, "b": self.state.b})
        )
        print("The sum is:", result.raw)
        try:
            self.state.num_result = int(result.raw.split()[-1])
        except (ValueError, IndexError):
            pass

def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
