import pytest
import re
from numsy.solver.calculus import differentiate

from numsy.parser import gts

problems = list(enumerate([x for x in open(r"tests/test_derivative.txt", encoding="UTF-8").readlines() if not x.startswith("#") and x != "\n"]))


@pytest.mark.parametrize("number, problem", problems)
def test_main(number, problem: str):
    problem, answer, optional_n = re.match(r"(.+?)\s*==\s*(.+?)(?:,\s*n\s*=\s*(\d+))?$", problem.replace(" ", "")).groups()
    solved = differentiate(problem, term="x", n=int(optional_n) if optional_n is not None else 1)
    result = gts(solved).replace(" ", "") == answer

    err_message = f"{number} | {problem}, got {gts(solved)} ❌ [Expected answer: {answer}]"
    assert result, err_message
