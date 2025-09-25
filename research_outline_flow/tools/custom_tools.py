import math

class PythagoreanTheorem:
    """
    A class to solve and explain problems related to the Pythagorean Theorem.
    """

    @staticmethod
    def calculate_hypotenuse(a: float, b: float) -> float:
        """
        Calculate the hypotenuse (c) of a right triangle given the two legs (a, b).
        Formula: c = sqrt(a^2 + b^2)
        """
        if a <= 0 or b <= 0:
            raise ValueError("Both sides 'a' and 'b' must be greater than 0.")
        c = math.sqrt(a**2 + b**2)
        return c

    @staticmethod
    def calculate_leg(c: float, known_leg: float) -> float:
        """
        Calculate a missing leg (a or b) of a right triangle given the hypotenuse (c)
        and one of the legs (known_leg).
        Formula: leg = sqrt(c^2 - known_leg^2)
        """
        if c <= 0 or known_leg <= 0 or c <= known_leg:
            raise ValueError("The hypotenuse must be greater than both legs.")
        missing_leg = math.sqrt(c**2 - known_leg**2)
        return missing_leg

    @staticmethod
    def verify_theorem(a: float, b: float, c: float) -> bool:
        """
        Verify if the sides satisfy the Pythagorean Theorem.
        Checks if: c^2 == a^2 + b^2
        """
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("All sides must be greater than 0.")
        return math.isclose(c**2, a**2 + b**2)

    @staticmethod
    def explain_theorem() -> str:
        """
        Provide a detailed explanation of the Pythagorean Theorem.
        """
        explanation = (
            "The Pythagorean Theorem states that in a right triangle, "
            "the square of the length of the hypotenuse is equal to the sum of the squares of the lengths "
            "of the other two sides. Mathematically, this is written as: c^2 = a^2 + b^2, where:\n"
            "- c is the hypotenuse (the longest side, opposite the right angle),\n"
            "- a and b are the two legs that form the right angle.\n\n"
            "To solve for the hypotenuse: c = sqrt(a^2 + b^2)\n"
            "To solve for a missing leg: a = sqrt(c^2 - b^2) or b = sqrt(c^2 - a^2)\n"
            "This theorem applies only to right-angled triangles."
        )
        return explanation

# Example Usage:

# Explaining the theorem
print(PythagoreanTheorem.explain_theorem())

# Calculating the hypotenuse
hypotenuse = PythagoreanTheorem.calculate_hypotenuse(3, 4)
print(f"Hypotenuse: {hypotenuse}")

# Calculating a missing leg
missing_leg = PythagoreanTheorem.calculate_leg(10, 6)
print(f"Missing Leg: {missing_leg}")

# Verifying the theorem
is_valid_triangle = PythagoreanTheorem.verify_theorem(6, 8, 10)
print(f"Is the triangle valid? {is_valid_triangle}")
