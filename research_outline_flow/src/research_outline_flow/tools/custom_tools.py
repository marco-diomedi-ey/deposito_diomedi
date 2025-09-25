@tool("my_simple_tool")
def my_simple_tool(params):
    class PythagoreanTheorem:
        def calculate_hypotenuse(self, a, b):
            return (a**2 + b**2) ** 0.5

        def calculate_leg(self, c, known_leg):
            if c > known_leg:
                return (c**2 - known_leg**2) ** 0.5
            else:
                raise ValueError("Hypotenuse must be greater than the known leg")
    
    # Example of usage:
    theorem = PythagoreanTheorem()
    
    input_data = params.split(",")  # Expected input format: "mode,a,b,c" e.g., "hypotenuse,3,4,None"
    mode = input_data[0].strip()
    
    if mode == "hypotenuse":
        a = float(input_data[1].strip())
        b = float(input_data[2].strip())
        result = theorem.calculate_hypotenuse(a, b)
    elif mode == "leg":
        c = float(input_data[1].strip())
        known_leg = float(input_data[2].strip())
        result = theorem.calculate_leg(c, known_leg)
    else:
        result = "Invalid mode specified! Please use 'hypotenuse' or 'leg'."
    
    return result