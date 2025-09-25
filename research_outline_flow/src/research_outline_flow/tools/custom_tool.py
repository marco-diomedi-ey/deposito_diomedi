from crewai_tools import tool

@tool("my_simple_tool")
def my_simple_tool(params):
    import math

    class Pythagoras:
        def calculate_hypotenuse(self, a, b):
            return math.sqrt(a ** 2 + b ** 2)

        def calculate_side(self, c, known_side):
            return math.sqrt(c ** 2 - known_side ** 2)

    result = Pythagoras()
    return result
