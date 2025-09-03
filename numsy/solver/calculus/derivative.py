import copy

from decimal import Decimal

from numsy.parser import parse_group
from numsy.parser.objects import Operator, Group, Fraction, ParenthesizedGroup

from numsy.solver.solve_algebra import simplify_side, multiply_all, get_common_factors
from numsy.solver.core import Positions
from numsy.solver.utility import clean_equation, convert_division_to_fraction
from numsy.solver.datatype import No_RO


def differentiate_single_group(group: Group, term: str, override_coefficient: Fraction = None) -> No_RO:
    # A group with no variable
    if not group.variable:  # TODO: Group with variable power
        return [Group()]

    # A normal group with a variable and numerical powers
    if group.power and not group.power_contains_variable and group.variable.name == term:
        position = Positions(group.power)
        power = simplify_side(position)[0]
        if override_coefficient:
            override_coefficient *= power
        else:
            group.number.value *= power.number.value
        power.number.value -= 1
        if power.number.value == 0:
            group.power = []
        else:
            group.power = [power]

    # A group with a variable but no power, then we can just delete the variable and make it a constant
    if not group.power and group.variable.name == term:
        group.variable = None

    # There's a leading constant (a Fraction)
    if override_coefficient:
        resolved = override_coefficient.resolve()
        if not isinstance(resolved, Group):
            return [override_coefficient, Operator.Mul, group]
        elif resolved.number.value == -1:
            group = group.invert_neg()
    return [group]


def perform_chain_rule(group: ParenthesizedGroup, term: str, leading_coefficient: Group = None):
    new = []
    if not group.contains_variable:  # TODO: Group with variable power
        return []

    # When constant in the numerator is not 1 OR there's a coefficient in the denominator, then we can change
    # the numerator and transformed to be a single value (fraction), which is (<NUMERATOR_COEF>/<DENO_COEF>)
    coefficient = Fraction(
        numerator=[leading_coefficient if leading_coefficient is not None else Group.from_value(Decimal(1))],
        denominator=[Group.from_value(Decimal(1))]
    ).resolve()

    origin_power_is_0: bool = False
    if group.power and not group.power_contains_variable:
        position = Positions(group.power)
        power_single = simplify_side(position)[0]
        coefficient *= power_single
        power_single.number.value -= 1
        if power_single.number.value == 0:
            origin_power_is_0 = True
        elif power_single.number.value == 1:
            group.power = []
        else:
            group.power = [power_single]

    # Prevent differentiation from affecting origin groups
    grpcpy = copy.deepcopy(group.groups)
    derivative = differentiate(grpcpy, term=term, _is_internal_calling=True)
    if coefficient != 1:
        # Chain rule form: <coef>*(<origin>)^<pow - 1> * <d/dx (<origin>)>
        new += [coefficient, *((Operator.Mul, group) if not origin_power_is_0 else []), Operator.Mul, ParenthesizedGroup(derivative)]
    else:
        new += [group, Operator.Mul, ParenthesizedGroup(derivative)]
    return new


def perform_differentiation(groups: No_RO, term: str):
    new: No_RO = []
    for i, group in enumerate(groups):
        if isinstance(group, Operator):
            # Delete the previous group if it's zero after differentiation
            if isinstance(g := new[-1], Group) and g.is_zero:
                del new[-1]
            else:
                new.append(group)
        elif isinstance(group, Group):
            if (
                (len(groups) == 1)  # Only 1 group
                or (i == 0 and groups[i + 1] == Operator.Addition)  # First item
                or (i == len(groups) - 1 and groups[i - 1] == Operator.Addition)  # Last item
                or (groups[i - 1] == Operator.Addition and groups[i + 1] == Operator.Addition)  # Middle item
            ):
                derivative = differentiate_single_group(group, term=term)
                new += derivative

        elif isinstance(group, Fraction):
            # This is the <CONSTANT>/x or <CONSTANT>/(PG) format
            if isinstance(group.numerator[0], Group) and not group.numerator[0].variable:
                if isinstance(group.denominator[0], Group):
                    transformed = group.denominator[0]
                    num_coef = group.numerator[0].number.value
                    coefficient = None
                    if num_coef != 1 or transformed.number.value != 1:
                        coefficient = Fraction(
                            numerator=[group.numerator[0]],
                            denominator=[Group.from_value(transformed.number.value)]
                        ).resolve()
                        transformed.number.value = 1
                    if transformed.power:
                        transformed.power = multiply_all(group.denominator[0].power, multiplier=-1)
                    else:
                        transformed.power = [Group.from_value(Decimal(-1))]
                    new += differentiate_single_group(transformed, term=term, override_coefficient=coefficient)
                elif isinstance(group.denominator[0], ParenthesizedGroup):
                    # TODO: This
                    ...
        elif isinstance(group, ParenthesizedGroup):
            new += perform_chain_rule(group, term=term)
    # Perform cleanup if the last item is zero
    if len(new) > 1 and isinstance(g := new[-1], Group) and g.is_zero:
        del new[-1]
        if new and new[-1] == Operator.Addition:
            del new[-1]
    return new


def differentiate(equation: No_RO | str, term: str, n: int = 1, _is_internal_calling: bool = False) -> No_RO:
    """Use `_is_internal_calling` to bypass input sanitization."""
    if isinstance(equation, str):
        groups = parse_group(equation)
    else:
        groups = equation
    if not _is_internal_calling:
        groups = convert_division_to_fraction(clean_equation(groups))

    res = groups
    for i in range(n):
        res = perform_differentiation(res, term=term)
    return res
