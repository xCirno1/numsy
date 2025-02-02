# NumSy
Welcome to NumSy, a project designed to ensure precise and accurate mathematical calculations. This is an independent project, which eliminates the need for any additional libraries. Our primary goal is to provide users with an easy-to-use platform for solving math problems. We highly welcome your feedback, suggestions, and contributions, so please don't hesitate to create issues or pull requests.

## Installation
To install NumSy using PyPI, run the following command:

    $ pip install numsy

## Basic Example
* #### As a calculator
```python
from numsy import solver

# Let's try to calculate triangle area with a height of 5
# and base length of 10
answer = solver.solve("1/2 * 10 * 5")
print(answer)  # Prints 25
```
* #### As a math solver
```python
from numsy import solver

answer = solver.solve("4x + 3 = 19")
print(answer.x)  # Prints 4
```

* #### Solving Matrix

```python
from numsy.solver import Matrix

m1 = [
    [1, 2, 3],
    [3, 2, 1],
    [0, 0, 8]
]

print(Matrix(m1).inverse().matrix)
# [[-0.5, 0.5, 0.125], [0.75, -0.25, -0.25], [-0.0, -0.0, 0.125]]

print(Matrix(m1).adjugate().matrix)
# [[16, -16, -4], [-24, 8, 8], [0, 0, -4]]
```

## Features
- Parsing problems and equations
- Safe calculation without the usage of `eval` or `exec`
- Basic arithmetics calculation (PEMDAS problem)
- Solving linear algebra (1 variable)
- Matrices

## How it works
NumSy operates by accepting problems and equations as strings and using its own algorithm to solve them. Here's a breakdown of the process:
1. NumSy receives the problem or equation as a string.
2. It parses the string character by character, identifying the type of each character (operator, digit, etc.).
3. Once the parsing phase is complete, NumSy determines the problem's identity, such as the number of variables or whether it's an equation.
4. It then categorizes the problem type, whether it falls under Basic PEMDAS, Quadratic Equation, or other categories.
5. Using specific functions bound to each problem type, NumSy proceeds to solve the problem.
