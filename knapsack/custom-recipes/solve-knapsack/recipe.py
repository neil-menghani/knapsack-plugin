import dataiku
from dataiku.customrecipe import *
from dataiku import pandasutils as pdu
import pandas as pd, numpy as np
import cvxpy

# Retrieve input and output dataset names
input_dataset_name = get_input_names_for_role('input_dataset')[0]
output_dataset_name = get_output_names_for_role('output_dataset')[0]

# Retrieve mandatory user-defined parameters
label_col = get_recipe_config()['label_col']
cost_col = get_recipe_config()['cost_col']
value_col = get_recipe_config()['value_col']
cap = get_recipe_config()['cap']
selection_n = get_recipe_config()['selection_n']

# Retrieve optional user-defined parameters
agg_col = get_recipe_config().get('agg_col', None)
actual_col = get_recipe_config().get('actual_col', None)
top_n = get_recipe_config().get('top_n', 1)

# Error checking of user-defined parameters


# Read input dataset as dataframe
input_dataset = dataiku.Dataset(input_dataset_name)
knapsack_df = input_dataset.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
agg_distinct = list(knapsack_df[agg_col].unique())
agg_distinct.sort()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Inspiration from: https://towardsdatascience.com/integer-programming-in-python-1cbdfa240df2

def knapsack(df, label_col, value_col, cost_col, selection_n, max_val):

    # gather elements to select from
    costs = np.array(df[cost_col])
    values = np.array(df[value_col])
    selection = cvxpy.Variable(len(costs), boolean=True)

    # objective function (to maximize)
    total_value = values @ selection

    # universal constraints (weight constraints and constraint on number of selections)
    cost_constraint = costs @ selection <= cap
    roster_constraint = np.ones(len(costs)) @ selection == selection_n
    max_constraint = values @ selection <= max_val # max val allows us to get "n best" optimal values

    # specific constraints (custom constraints for situation)
    pg = np.array(df['PG'])
    sg = np.array(df['SG'])
    sf = np.array(df['SF'])
    pf = np.array(df['PF'])
    c = np.array(df['C'])
    pg_constraint = pg @ selection >= 1
    sg_constraint = sg @ selection >= 1
    sf_constraint = sf @ selection >= 1
    pf_constraint = pf @ selection >= 1
    c_constraint = c @ selection >= 1
    g_constraint = (pg + sg) @ selection >= 3
    f_constraint = (sf + pf) @ selection >= 3

    knapsack_problem = cvxpy.Problem(cvxpy.Maximize(total_value), [cost_constraint, roster_constraint, max_constraint,
                                                                   pg_constraint, sg_constraint, sf_constraint,
                                                                   pf_constraint, c_constraint, g_constraint, f_constraint])
    value_opt = knapsack_problem.solve(solver=cvxpy.GLPK_MI)
    selection_opt = np.array(selection.value).astype(int)
    total_cost = (costs @ selection).value

    return selection_opt, value_opt, total_cost

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
output_data = []
for agg_bin in agg_distinct:
    print(agg_bin)
    max_val = np.inf
    for iteration in range(top_n):
        knapsack_bin_df = knapsack_df.loc[knapsack_df[agg_col] == agg_bin]
        selection_opt, value_opt, total_cost = knapsack(knapsack_bin_df, label_col, value_col, cost_col, selection_n, max_val)

        labels, actuals = list(knapsack_bin_df[label_col]), list(knapsack_bin_df[actual_col])
        labels_opt, actuals_opt = [], []
        for i in range(len(selection_opt)):
            if selection_opt[i]:
                labels_opt.append(labels[i])
                actuals_opt.append(actuals[i])
        output_row = [agg_bin]
        output_row.extend(labels_opt)
        output_row.extend([int(total_cost), np.round(value_opt, 3), np.sum(actuals_opt)])
        output_data.append(output_row)
        max_val = value_opt - 10**-3

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
output_cols = [agg_col]
output_cols.extend(range(1, selection_n+1))
output_cols.extend(['Total_Cost', 'Value_Predict', 'Value_Actual'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
nba_knapsack_output_df = pd.DataFrame(output_data, columns=output_cols)
nba_knapsack_output_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
nba_knapsack_output = dataiku.Dataset(output_dataset_name)
nba_knapsack_output.write_with_schema(nba_knapsack_output_df)