# import packages
import numpy as np
import pandas as pd
import os
from itertools import product
from collections.abc import Iterable
import math


def allowed_output(value, reaction_vol_nl=20000, drop_size_nl=100, verbose=0):
    """Based on high ,low and stock concentrations and droplet size calculate how many combinations is possible

    Parameters
    ----------
    value: tuple
        (low, high, stock concentration)

    Returns
    -------
    calculated_concs:
        a list of possible concentrations
    
    calculated_vols:
        a list of possible volumes
    """

    if value['Conc_Values']:
        if isinstance(value['Conc_Stock'], Iterable):
            drop_nums = [i * reaction_vol_nl / (drop_size_nl * value['Conc_Stock'][find_stock(value['Conc_Values'], value['Conc_Stock'], i)[0]]) for i in value['Conc_Values']]
            calculated_concs = value['Conc_Values']
        else:
            drop_nums = [i * reaction_vol_nl / (drop_size_nl * value['Conc_Stock']) for i in value['Conc_Values']]
            calculated_concs = value['Conc_Values']

    else:
        drop_nums = list(range(int((value['Conc_Min'] * reaction_vol_nl) / (drop_size_nl * value['Conc_Stock'])),
                               int((value['Conc_Max'] * reaction_vol_nl) / (drop_size_nl * value['Conc_Stock'])) + 1))

        calculated_concs = [drop_num * value['Conc_Stock'] * drop_size_nl / reaction_vol_nl for drop_num in drop_nums]

    if verbose:
        print('drops :', drop_nums)
        print('volumes :', [i * drop_size_nl for i in drop_nums])
        print('possible_concentrations :', calculated_concs)
    else:
        return calculated_concs, [i * drop_size_nl for i in drop_nums]


def percentage_possible(data, threshold=40):
    """Based on threshold volume, it calculates how many combinations of all metabolite is possible to make

    Parameters
    ----------
    data: dict
        {'meatbolite name':[possible volumes], ...}
        
    Returns
    -------
    percentage possible: float    
    total: int
        total number of combinations (includes forbidden one)
    """
    lists = list(data.values())

    m = [len(i) for i in data.values()]

    total = np.prod(np.array([len(i) for i in data.values()]))
    possible = 0

    for items in product(*lists):
        if sum(items) <= threshold:
            possible += 1
    
    return (possible/total*100), total

def find_stock(conc_values, conc_stocks, this_value):
    """this function find each concentration value belongs to wich stock concentration for metabolites with multiple stocks

    Parameters
    ----------
    conc_values: list
        a list of all possible concentration
        
    conc_stocks: list
        a list of all stocks concentration
        
    this_value: float, int
        concentration value that we find to find its stock
        
    Returns
    -------
    i:
        index of found stock
        
    out:
        value of found stock
    """
    num = len(conc_stocks)
    avg = len(conc_values) / float(num)
    out = []
    last = 0.0

    while last < len(conc_values):
        out.append(conc_values[int(last):int(last + avg)])
        last += avg

    for i, value in enumerate(out):
        if this_value in value:
            return i, out

# random combination generator function_v3.0
def random_combination_generator(concentrations_limits, number_of_combination=100, reaction_vol_nl=10000,
                                 max_nl=None, drop_size_nl=100, check_repeat=True, rounded=2, verbose=0, make_csv=False, return_df=False):
    """this function make random combination that is safe (e.g. dont make too much or low concentrated, not excecutable based on drop size, not repetitive)

    Parameters
    ----------
    concentrations_limits: dict
        {'name of metabolite': {'Conc_Min': #, 'Conc_Max': #, 'Conc_Values': #, 'Conc_Stock': #, 'Alternatives': #}, ...}
        
    Returns
    -------
    data: pandas.DataFrame
        a dataframe as consists of number_of_combination of random combinations
    """
    
    # generating random combinations
    combinations = []
    data_point = 0
    while data_point < number_of_combination:
        input_data = []
        input_vol = []
        # verbosity
        if (data_point % 10000 == 0) and verbose:
            print(data_point)

        # generation of random input
        for key, value in concentrations_limits.items():
            # Manual Concentration Value Generation
            if value['Conc_Values']:
                # With Alternatives
                if value['Alternatives']:
                    num_alternative = len(value['Alternatives'])
                    choice_alternative = np.random.randint(0, num_alternative)
                    choice_list = [0 for i in range(num_alternative)]
                    choice_list[choice_alternative] = 1

                    choice_conc = np.random.choice(value['Conc_Values'])
                    input_data.append(choice_conc)
                    input_data += choice_list
                    if isinstance(value['Conc_Stock'], Iterable):
                        choice_stock, _ = find_stock(value['Conc_Values'], value['Conc_Stock'], choice_conc)
                        input_vol.append(choice_conc/value['Conc_Stock'][choice_stock]*reaction_vol_nl)
                    else:
                        input_vol.append(choice_conc/value['Conc_Stock']*reaction_vol_nl)

                # Without Alternatives
                else:
                    choice_conc = np.random.choice(value['Conc_Values'])
                    input_data.append(choice_conc)
                    if isinstance(value['Conc_Stock'], Iterable):
                        choice_stock, _ = find_stock(value['Conc_Values'], value['Conc_Stock'], choice_conc)
                        input_vol.append(choice_conc/value['Conc_Stock'][choice_stock]*reaction_vol_nl)
                    else:
                        input_vol.append(choice_conc/value['Conc_Stock']*reaction_vol_nl)

            # Auto Concentration Value Generation
            else:
                # With Alternatives
                if value['Alternatives']:
                    num_alternative = len(value['Alternatives'])
                    choice_alternative = np.random.randint(0, num_alternative)
                    choice_list = [0 for i in range(num_alternative)]
                    choice_list[choice_alternative] = 1

                    drop_num = np.random.randint(round(value['Conc_Min'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']),
                                                 round(value['Conc_Max'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']) + 1)

                    recalculated_conc = drop_num * value['Conc_Stock'] * drop_size_nl / reaction_vol_nl
                    input_data.append(recalculated_conc)
                    input_data += choice_list
                    input_vol.append(recalculated_conc/value['Conc_Stock']*reaction_vol_nl)

                # Without Alternatives
                else:
                    drop_num = np.random.randint(round(value['Conc_Min'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']),
                                                 round(value['Conc_Max'] * (reaction_vol_nl / drop_size_nl) / value['Conc_Stock']) + 1)

                    recalculated_conc = drop_num * value['Conc_Stock'] * drop_size_nl / reaction_vol_nl
                    input_data.append(recalculated_conc)
                    input_vol.append(recalculated_conc/value['Conc_Stock']*reaction_vol_nl)
        
        # Checks
        if check_repetitive and max_nl:
            if input_data not in combinations and sum(input_vol)<= max_nl:
                combinations.append(input_data)
                data_point += 1
        elif check_repetitive and not max_nl:
            if input_data not in combinations:
                combinations.append(input_data)
                data_point += 1
        elif not check_repetitive and max_nl:
            if sum(input_vol)<= max_nl:
                combinations.append(input_data)
                data_point += 1
        else:
            combinations.append(input_data)
            data_point += 1

    # make column name:
    columns_name = []
    for key, value in concentrations_limits.items():
        if not value['Alternatives']:
            columns_name.append(key)
        else:
            columns_name.append(key)
            alternative_name = ['{}_{}'.format(key, i) for i in value['Alternatives']]
            columns_name += alternative_name

    # making csv file
    if make_csv:
        data = pd.DataFrame(np.array(combinations), columns=columns_name)
        data.to_csv('Random_Combination_1.csv', index=False)

    # making dataframe
    if return_df:
        data = pd.DataFrame(np.array(combinations), columns=columns_name)
        return data

    return np.array(combinations)

# transform concentration DataFrame to volume (nanolitre) DataFrame
def concentration_to_volume(concentrations, concentrations_limits, reaction_mixture_vol_nl=10000,
                            fixed_parts={'Lysate': 0.33, 'Saline': 0.1}, round_deg=1, check_water=True, minimum_drop_size_nanoliter=25):
    """Transform concentrations dataframe to volumes dataframe
       option: add a fixed volumes to all combinations like Lysate
       caution: concentrations unit and metabolite name in concentrations and concentrations_limits must be the same.

    Parameters
    ----------
    concentrations: pandas.DataFrame
        random_combination_generator output
    
    Returns
    -------
    data: pandas.DataFrame
        a dataframe same as input in shape but volumes data
    """

    # make a copy of original dataframe to avoid further change than can affect that
    data = concentrations.copy(deep=True)
    data_all = data.copy(deep=True)
    data = data[[i for i in data.columns if '_' not in i]]
    data *= reaction_mixture_vol_nl

    for metabolite_name, value in concentrations_limits.items():
        if isinstance(value['Conc_Stock'], Iterable):
            print()
            data[metabolite_name] = [round(data[metabolite_name][i] / value['Conc_Stock'][find_stock(value['Conc_Values'], value['Conc_Stock'], data_all[metabolite_name][i])[0]], round_deg) for i in range(len(data[metabolite_name]))]
        else:
            data[metabolite_name] = [round(data[metabolite_name][i] / value['Conc_Stock'], round_deg) for i in range(len(data[metabolite_name]))]

    # add fix parts
    if fixed_parts:
        for key, value in fixed_parts.items():
            data[key] = reaction_mixture_vol_nl * value

    # add water to reach the reaction_mixture_vol_nl
    data['water'] = reaction_mixture_vol_nl - data.sum(axis=1)

    # for low stock concentration that is not possible to make, raise an error
    # stock conc should be set in a way that doesn't raise this error to avoid further debugging
    if check_water and not all(data['water'] >= 0): raise Exception("Oops, too concentrated combination!")

    # Check for minimum drop size
    for column in data.columns:
        if (data[column] < minimum_drop_size_nanoliter).any():
            # Exclude zero values from the exception check
            if (data[column][data[column] != 0] < minimum_drop_size_nanoliter).any():
                raise Exception(f"Volume for item {column} is below the minimum drop size of {minimum_drop_size_nanoliter} nanoliters!")


    # add alternative
    # make columns name list:
    columns_name = []
    Type_dic = {}
    Stock_dic = {}
    for key, value in concentrations_limits.items():
        if value['Alternatives']:
            columns_name.append(key)
            columns_name.append('{}_Type'.format(key))
            Type_dic[key] = []
        else:
            columns_name.append(key)
        if isinstance(value['Conc_Stock'], Iterable):
            columns_name.append('{}_Stock_Type'.format(key))
            Stock_dic[key] = []

    # Alternatives
    for key in Type_dic.keys():
        data_type = data_all[[i for i in data_all.columns if '{}_'.format(key) in i]]
        for i in data_type.values:
            Type_dic[key].append(concentrations_limits[key]['Alternatives'][np.where(i == 1.0)[0][0]])

    Type_list = list(Type_dic.keys())
    for key in Type_list:
        Type_dic['{}_Type'.format(key)] = Type_dic.pop(key)

    # Stock
    for key in Stock_dic.keys():
        Stock_dic[key] = [concentrations_limits[key]['Conc_Stock'][find_stock(concentrations_limits[key]['Conc_Values'], concentrations_limits[key]['Conc_Stock'], i)[0]] for i in data_all[key]]
         
    Stock_list = list(Stock_dic.keys())
    for key in Stock_list:
        Stock_dic['{}_Stock_Type'.format(key)] = Stock_dic.pop(key)

    data_final = pd.concat([data, pd.DataFrame(Type_dic), pd.DataFrame(Stock_dic)], axis=1)
    return data_final[columns_name + list(fixed_parts.keys()) + ['water']]


def day_finder(file, file_format='csv'):
    """Find the first notcompleted day

    Parameters
    ----------
    file: 
        for now, it can only be 'Results'
        
    Returns
    -------
    i: int
        the first not completed day
    """
    i = 1
    while True:
        if not os.path.isfile('{}_{}.{}'.format(file, i, file_format)):
            return i
        i += 1


def result_preprocess(day, desired_cols, range=20):
    """Preprocess Results.csv file to get desired columns and rows
        caution: the target column name MUST be 'yield'
        
    Parameters
    ----------
    day: 
        Results_day.csv
    
    desired_cols:
        name of columns that you want from the results file
        
    Returns
    -------
    data_m:
        data in range
    label_m:
        label in range
    data_specials:
        other data
    label_specials:
        other labels
    """
    results = pd.read_csv('Results_{}.csv'.format(day, day))

    # m number pipeline
    data_m = results[desired_cols].iloc[:range, :]
    label_m = results[['yield']].iloc[:range, :]

    # reference, control and specials
    data_specials = results[desired_cols].iloc[range:, :]
    label_specials = results[['yield']].iloc[range:, :]

    return data_m, label_m, data_specials, label_specials


def check_repetitive(combination, df_main):
    """Check to avoid repetitive combinations
        
    Parameters
    ----------
    combination: 
        combinations that want to be checked
    
    df_main: pandas.DataFrame
        source dataframe
        
    Returns
    -------
    boolean:
        True: it exists in df_main
        False: it's not
    """
    comparison_df = df_main.merge(combination, indicator=True, how='outer')
    if 'both' in comparison_df._merge.unique():
        return False
    else:
        return True


def bayesian_optimization(regressors_list,
                          data, label,
                          concentrations_limits,
                          final_order,
                          df_main,
                          reaction_vol_nl=20000, max_nl=13200, drop_size_nl=100,
                          exploitation=1, exploration=1, test_size=100, pool_size=100000, verbose=0, day=1,
                          days_range=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                          batch_ucb=False):
    """Main bayesian optimization function
        
    Parameters
    ----------
    regressors_list: 
        a list consists of more than one regressor that has .fit and .predict feature
    
    data: pandas.DataFrame
        all previous day data

    label: pandas.DataFrame
        all previous day label
        
    exploitation: 1
        coefficient of focus on higher yield query
    
    exploration: 1
        coefficient of focus on a more informative query
        
    test_size: 100
        output combinations number
        
    pool_size: 100000
        how many random combinations to ask from the regressor list each round
        caution: this parameter highly affects executions time
        
    Returns
    -------
    chosen_combinations: pandas.DataFrame
        combinations that expected to improve yield

    if batch_ucb == True
    Returns
    -------
    best sample based on ucb: pandas.Series, best sample's expected value: float
    """
    # first fit training data on our models
    for regressor in regressors_list:
        regressor.fit(data.values, label.values)

    # make random test data
    df_1 = random_combination_generator(concentrations_limits, number_of_combination=pool_size,
                                        reaction_vol_nl=reaction_vol_nl,
                                        max_nl=max_nl, drop_size_nl=drop_size_nl, make_csv=False, return_df=True)
    desired_cols = list(df_1.columns)

    df_temp = df_1.copy(deep=True)

    # Upper Confidence Bound
    for index, regressor in enumerate(regressors_list):
        df_1['pred_yield_{}'.format(index)] = regressor.predict(df_temp.values)

    df_1['regressors_std'] = df_1[[str(i) for i in df_1.columns if 'pred_yield' in str(i)]].std(axis=1)
    df_1['mean_vote'] = df_1[[str(i) for i in df_1.columns if 'pred_yield' in str(i)]].mean(axis=1)
    df_1['UCB'] = exploitation * df_1['mean_vote'] + exploration * df_1['regressors_std']
    df_1 = df_1.sort_values(['UCB'], ascending=False)

    if batch_ucb:
        return df_1[final_order].iloc[0:1, :], df_1['mean_vote'].values[0]
    # check to don`t make repeated combinations, but it is not likely

    chosen_combinations = pd.DataFrame(columns=desired_cols)
    num = 0
    for i in df_1[desired_cols].values:
        temp_combination = pd.DataFrame([i], columns=desired_cols)
        if check_repetitive(temp_combination, df_main):
            num += 1
            chosen_combinations = pd.concat([chosen_combinations, temp_combination]).reset_index(drop=True)
        if num == test_size:
            break

    return chosen_combinations[final_order]


# Batch UCB on top of bayesian optimization
def batch_ucb(regressors_list,
                data, label,
                concentrations_limits,
                final_order,
                df_main,
                reaction_vol_nl=20000, max_nl=13200, drop_size_nl=100,
                exploitation=1, exploration=1, test_size=100, pool_size=100000, verbose=0, day=1,
                days_range=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20]):
    """Batch UCB on top of bayesian optimization function
        
    Parameters
    ----------
    regressors_list: 
        a list consists of more than one regressor that has .fit and .predict feature
    
    data: pandas.DataFrame
        all previous day data

    label: pandas.DataFrame
        all previous day label
        
    exploitation: 1
        coefficient of focus on higher yield query
    
    exploration: 1
        coefficient of focus on a more informative query
        
    test_size: 100
        output combinations number
        
    pool_size: 100000
        how many random combinations to ask from the regressor list each round
        caution: this parameter highly affects executions time
        
    Returns
    -------
    chosen_combinations: pandas.DataFrame
        combinations that expected to improve yield
    """
    
    final_samples = []

    for i in range(test_size):
        sample, expected_value = bayesian_optimization(regressors_list, data, label, concentrations_limits,
                                            final_order=final_order,
                                            df_main = df_main,
                                            reaction_vol_nl=reaction_vol_nl, max_nl=max_nl,
                                            drop_size_nl=drop_size_nl,
                                            exploitation=exploitation, exploration=exploration, test_size=test_size, pool_size=pool_size, verbose=0, day=day, days_range = days_range,
                                            batch_ucb=True)
        final_samples.append(sample)
        data = pd.concat([data, sample], axis=0).reset_index(drop=True)
        label = pd.concat([label, pd.DataFrame({'yield': [expected_value]})], axis=0).reset_index(drop=True)

    return pd.concat(final_samples)    

from logging import error
# ECHO functions

def put_volumes_to_384_wells(volumes_array, starting_well='B2', vertical=False, make_csv=False):
    """Make a dataframe as a 384 well plate for each metabolite
        
    Parameters
    ----------
    volumes_array: 
        a dataframe with columns are component, each row vol of that component (e.g. volumes.csv) 
        
    starting_well: 'A1'
        name of the well in 384 well plates that you want to start filling
    
    vertical:
        if True, it will fill the plate column by column top down
        if False, it will fill the plate row by row, left to right
        
    Returns
    -------
    all_dataframe:
        a list consists of one dataframe for each metabolite that shows appropriate 384 well plate
        
    named_volumes:
        one separate dataframe that adds well name to volume dataframe
    """
    if len(volumes_array) > 294: raise ValueError

    all_dataframe = {}
    rows_name = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']

    if not vertical:
        from_well = rows_name.index(starting_well[0]) * 22 + int(starting_well[1:]) - 1
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 25))
            # add data one by one in each row
            # (0, 0)--------->(0,23)
            # .......................
            # (15,0)--------->(15,23)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index // 22, index % 22] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[index // 22], index % 22 + 1) for index in named_volumes.index]
        named_volumes['well_name'] = names

    if vertical:
        from_well = rows_name.index(starting_well[0]) + (int(starting_well[1:]) - 1) * 14
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 25))
            # add data one by one in each column
            # (0, 0)---->-----(0,23)
            # ||||||..........||||||
            # (15,0)---->-----(15,23)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index % 14, index // 14] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[(index + from_well) % 14], (index + from_well) // 14 + 1) for index in
                 named_volumes.index]
        named_volumes['well_name'] = names

    # notice that this function output two value
    return named_volumes, all_dataframe

def put_volumes_to_96_wells(volumes_array, starting_well='A1', vertical=False, make_csv=False):
    """Make a dataframe as a 96 well plate for each metabolite
        
    Parameters
    ----------
    volumes_array: 
        a dataframe with columns are component, each row vol of that component (e.g. volumes.csv) 
        
    starting_well: 'A1'
        name of the well in 96 well plates that you want to start filling
    
    vertical:
        if True, it will fill the plate column by column top down
        if False, it will fill the plate row by row, left to right
        
    Returns
    -------
    all_dataframe:
        a list consists of one dataframe for each metabolite that shows appropriate 384 well plate
        
    named_volumes:
        one separate dataframe that adds well name to volume dataframe
    """
    if len(volumes_array) > 96: raise ValueError

    all_dataframe = {}
    rows_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    if not vertical:
        from_well = rows_name.index(starting_well[0]) * 12 + int(starting_well[1:]) - 1
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 13))
            # add data one by one in each row
            # (0, 0)--------->(0,11)
            # .......................
            # (7,0)--------->(7,11)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index // 12, index % 12] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[index // 12], index % 12 + 1) for index in named_volumes.index]
        named_volumes['well_name'] = names

    if vertical:
        from_well = rows_name.index(starting_well[0]) + (int(starting_well[1:]) - 1) * 8
        # make each metabolite`s dataframe and add it to dict
        for metabolite_name in volumes_array.columns:
            # first make an all zero dataframe
            dataframe = pd.DataFrame(0.0, index=rows_name, columns=range(1, 13))
            # add data one by one in each column
            # (0, 0)---->-----(0,11)
            # ||||||..........||||||
            # (7,0)---->-----(7,11)
            for index, value in enumerate(volumes_array[metabolite_name]):
                index += from_well
                dataframe.iloc[index % 8, index // 8] = value

            all_dataframe[metabolite_name] = dataframe

        # make named volumes dataframe
        named_volumes = volumes_array.copy(deep=True)
        names = ['{}{}'.format(rows_name[(index + from_well) % 8], (index + from_well) // 8 + 1) for index in
                 named_volumes.index]
        named_volumes['well_name'] = names

    # include code to raise error if there are not enough wells to accomodate all combinations and suggest to use 384 wells.

    # notice that this function output two value
    return named_volumes, all_dataframe


def check_wells_capacity(volumes_array, triplicate=False, destination_plate_384_well=True, vertical=True):
    """
    This function checks which destination plate (384 or 96) is needed to perform experiments required.

    Parameters
    ----------
    volumes_array: DataFrame
        DataFrame where each column is a component and each row is the volume of that component.
        
    triplicate: bool
        If True, calculates wells needed for triplicates (default is False).
        
    destination_plate_384_well: bool
        If True, checks capacity for a 384-well plate. If False, checks for a 96-well plate (default is True).
    Raises
    ------
    ValueError
        If the number of wells needed exceeds the capacity of the specified plate.
    """

    # Check the number of wells needed
    num_combinations_per_round = volumes_array.shape[0]
    if triplicate:
        wells_needed = num_combinations_per_round * 3
    else:
        wells_needed = num_combinations_per_round

    if wells_needed > 96 and not destination_plate_384_well:
          raise ValueError(f"Error: The number of wells needed ({wells_needed}) exceeds the capacity of a 96-well plate. Use 384-well plate by setting destination_plate_384_well = True.")

    if vertical:
      # 384-well plate avoids using the most outer wells for reliability of the ECHO liquid handler.
      if wells_needed > 294:
          raise ValueError(f"Error: The number of wells needed ({wells_needed}) exceeds the capacity of a 384-well plate. Consider reducing the number of combinations or avoiding triplicates.")

    else:
      if wells_needed > 264:
          raise ValueError(f"Error: The number of wells needed ({wells_needed}) exceeds the capacity of a 96-well plate. Use 384-well plate by setting destination_plate_384_well = True.If the number of combination is between 264 and 294, you can use 384 plate vertically by vertical=True")

    
    return "Destination Well Capacity is sufficient for this experiment."



def put_volumes_to_wells(volumes_array, destination_plate_384_well=True, vertical=True, triplicate=False, starting_well='A1', make_csv=False):
    """Helper function to allocate volumes into 96 or 384 well destination plates.

    Parameters
    ----------
    volumes_array: 
        DataFrame where columns are components and each row is the volume of that component (e.g., volumes.csv).
        
    starting_well: str
        Name of the well to start filling (default is 'A1').
    
    vertical: bool
        If True, fills the plate column by column top down. If False, fills row by row, left to right.
        
    Returns
    -------
    destination: DataFrame
        DataFrame that includes well names added to volume data.
    """

    # Check if the number of wells needed is acceptable
    check_wells_capacity(volumes_array, triplicate=triplicate, destination_plate_384_well=destination_plate_384_well)
    
    if destination_plate_384_well:
        if not triplicate:
            # Assign destination well for each combination on a 384-well plate.
            destination, _ = put_volumes_to_384_wells(volumes_array, starting_well=starting_well, vertical=vertical, make_csv=make_csv)
        else:
            # Assign destination well for each combination and for each replicate on a 384-well plate.
            # Use different starting wells based on vertical orientation.
            if vertical:
                replicate_1, _ = put_volumes_to_384_wells(volumes_array, starting_well='B2', vertical=vertical, make_csv=make_csv)
                replicate_2, _ = put_volumes_to_384_wells(volumes_array, starting_well='B9', vertical=vertical, make_csv=make_csv)
                replicate_3, _ = put_volumes_to_384_wells(volumes_array, starting_well='B16', vertical=vertical, make_csv=make_csv)
            else:
                replicate_1, _ = put_volumes_to_384_wells(volumes_array, starting_well='B2', vertical=vertical, make_csv=make_csv)
                replicate_2, _ = put_volumes_to_384_wells(volumes_array, starting_well='G2', vertical=vertical, make_csv=make_csv)
                replicate_3, _ = put_volumes_to_384_wells(volumes_array, starting_well='L2', vertical=vertical, make_csv=make_csv)
            destination = pd.concat([replicate_1, replicate_2, replicate_3]).reset_index(drop=True)
    else:
        if not triplicate:
          # Assign destination well for each combination on a 96-well plate.
          destination, _ = put_volumes_to_96_wells(volumes_array, starting_well=starting_well, vertical=vertical, make_csv=make_csv)
        else:
          # Assign destination well for each combination and for each replicate on a 96-well plate.
          # Use different starting wells based on vertical orientation.
          if vertical:
              replicate_1, _ = put_volumes_to_96_wells(volumes_array, starting_well='A1', vertical=vertical, make_csv=make_csv)
              replicate_2, _ = put_volumes_to_96_wells(volumes_array, starting_well='A5', vertical=vertical, make_csv=make_csv)
              replicate_3, _ = put_volumes_to_96_wells(volumes_array, starting_well='A9', vertical=vertical, make_csv=make_csv)
          else:
              replicate_1, _ = put_volumes_to_96_wells(volumes_array, starting_well='A2', vertical=vertical, make_csv=make_csv)
              replicate_2, _ = put_volumes_to_96_wells(volumes_array, starting_well='D2', vertical=vertical, make_csv=make_csv)
              replicate_3, _ = put_volumes_to_96_wells(volumes_array, starting_well='G2', vertical=vertical, make_csv=make_csv)
          destination = pd.concat([replicate_1, replicate_2, replicate_3]).reset_index(drop=True)
    
    return destination

def condense_source_wells(source_wells_df):
    """
    Make a more readable version of source well allocation table for easier pipetting.

    Parameters
    ----------
    source_wells_df: DataFrame
        DataFrame that includes source wells with volumes.

    Returns
    -------
    condensed_df: DataFrame
        DataFrame with condensed source wells.
    """
    condensed_wells = []
    current_item = None
    current_start_well = None
    current_end_well = None
    current_volume = 0

    def well_position(well):
        row = well[0]
        col = int(well[1:])
        return (row, col)

    def wells_are_contiguous(well1, well2):
        row1, col1 = well_position(well1)
        row2, col2 = well_position(well2)
        return row1 == row2 and col2 == col1 + 1

    for i, row in source_wells_df.iterrows():
        item = row['Item']
        well = row['Source Well']
        volume = row['Volume']

        if current_item is None:
            # Initialize the first group
            current_item = item
            current_start_well = well
            current_end_well = well
            current_volume = volume
        else:
            if item == current_item and wells_are_contiguous(current_end_well, well):
                # Continue the current group
                current_end_well = well
                current_volume += volume
            else:
                # Save the current group and start a new one
                if current_start_well == current_end_well:
                    well_range = current_start_well
                else:
                    well_range = f"{current_start_well}-{current_end_well}"
                condensed_wells.append({
                    'Item': current_item,
                    'Well Range': well_range,
                    'Volume': current_volume
                })
                current_item = item
                current_start_well = well
                current_end_well = well
                current_volume = volume

    # Add the last group
    if current_start_well == current_end_well:
        well_range = current_start_well
    else:
        well_range = f"{current_start_well}-{current_end_well}"
    condensed_wells.append({
        'Item': current_item,
        'Well Range': well_range,
        'Volume': current_volume
    })

    condensed_df = pd.DataFrame(condensed_wells)
    return condensed_df

# make source to destination dataframe for ECHO machine
def source_to_destination(named_volumes, desired_order=None, reset_index=True, check_zero=False):
    """Make a dataframe as a 384/96 well plate for each metabolite
        
    Parameters
    ----------
    named_volume: 
         first output of put_volumes_to_384_wells or put_volumes_to_96_wells function
        
    Returns
    -------
    all_sources:
        separate dataframe for each metabolite that appended in a dict
    
    aggregated:
        aggregated all_sources to one CSV file by your desired order
    """
    all_sources = {}
    for metabolite_name in named_volumes.drop(columns=['well_name']):
        transfers = {'Source_Plate_Barcode': [], 'Source_Well': [], 'Destination_Plate_Barcode': [],
                     'Destination_Well': [], 'Transfer_Volume': []}
        for index in range(len(named_volumes)):
            if named_volumes.loc[index, metabolite_name] > 0 or check_zero == False:
                transfers['Source_Plate_Barcode'].append('Plate1')
                transfers['Source_Well'].append('{} well'.format(metabolite_name))
                transfers['Destination_Plate_Barcode'].append('destPlate1')
                transfers['Destination_Well'].append(named_volumes.loc[index, 'well_name'])
                transfers['Transfer_Volume'].append(named_volumes.loc[index, metabolite_name])
        transfers = pd.DataFrame(transfers)

        all_sources[metabolite_name] = transfers

    # aggregate all dataframe
    aggregated = pd.concat(all_sources.values())

    if desired_order:
        aggregated = pd.concat([all_sources[i] for i in desired_order])

    if reset_index:
        aggregated = aggregated.reset_index(drop=True)

    return all_sources, aggregated

def create_ECHO_transfer_table(destination_df, source_wells_df, final_reaction_volume_nanoliter, minimum_drop_size_nanoliter, fixed_parts=None, Master_Mix_for_fixed_parts=False):
    """
    Create a transfer table for ECHO machine from source wells to destination wells.

    Parameters
    ----------
    destination_df: DataFrame
        DataFrame that includes destination wells with volumes (output from put_volumes_to_wells).

    source_wells_df: DataFrame
        DataFrame that includes source wells with volumes (output from calculate_source_wells).

    minimum_drop_size_nanoliter: int
        Minimum drop size that can be transferred.

    fixed_parts: dict
        Dictionary of fixed parts with their corresponding proportions.

    Master_Mix_for_fixed_parts: bool
        If True, delete rows corresponding to items in fixed_parts and summarize their total volumes.

    Returns
    -------
    ECHO_transfer_table_df: DataFrame
        DataFrame that includes item, source well, destination well, and transfer volume.
    
    fixed_parts_volumes: dict
        Dictionary summarizing the volumes of each fixed part to be added to each destination well.
    """
    # Initialize the transfer table
    transfer_table = []
    fixed_parts_volumes = {item: 0 for item in fixed_parts.keys()} if fixed_parts else {}

    # Iterate through each item.
    for item in destination_df.columns:
        if item == 'well_name':
            continue

        # Get the source wells for the current item
        source_rows = source_wells_df[source_wells_df['Item'] == item].reset_index(drop=True)

        # Get the destination wells for the current item
        destination_rows = destination_df[['well_name', item]].copy()
        destination_rows.columns = ['Destination Well', 'Transfer Volume']
        destination_rows = destination_rows[destination_rows['Transfer Volume'] > 0].reset_index(drop=True)

        # Create the transfer rows
        source_index = 0
        source_volume_left = source_rows.iloc[source_index]['Volume']

        # Iterate through all destination wells for this item.
        # If there is enough volume in the source well, transfer at once.
        # If not, transfer all remaining volume of the current source well and use the next source well to complete the transfer.
        for _, dest_row in destination_rows.iterrows():
            transfer_volume = dest_row['Transfer Volume']
            while transfer_volume > 0:
                if transfer_volume <= source_volume_left:
                    transfer_table.append({
                        'Item': item,
                        'Source Well': source_rows.iloc[source_index]['Source Well'],
                        'Destination Well': dest_row['Destination Well'],
                        'Transfer Volume': transfer_volume
                    })
                    source_volume_left -= transfer_volume
                    if source_volume_left < minimum_drop_size_nanoliter:
                        source_volume_left = 0
                    transfer_volume = 0
                else:
                    transfer_table.append({
                        'Item': item,
                        'Source Well': source_rows.iloc[source_index]['Source Well'],
                        'Destination Well': dest_row['Destination Well'],
                        'Transfer Volume': source_volume_left
                    })
                    transfer_volume -= source_volume_left
                    source_index += 1
                    if source_index < len(source_rows):
                        source_volume_left = source_rows.iloc[source_index]['Volume']
                    else:
                        raise ValueError(f"Not enough volume available in source wells for item {item}.")

    # Create DataFrame and remove rows with zero transfer volume
    ECHO_transfer_table_df = pd.DataFrame(transfer_table)
    ECHO_transfer_table_df = ECHO_transfer_table_df[ECHO_transfer_table_df['Transfer Volume'] > 0]
    ECHO_transfer_table_df = ECHO_transfer_table_df.reset_index(drop=True)

    # Optionally delete rows corresponding to items in fixed_parts and summarize volumes for each destination well
    if Master_Mix_for_fixed_parts and fixed_parts:
      # Create a new dictionary by multiplying each value in fixed_parts by final_reaction_volume_nanoliter
      fixed_parts_volumes = {key: value * final_reaction_volume_nanoliter for key, value in fixed_parts.items()}
        


    return ECHO_transfer_table_df, fixed_parts_volumes

# This function assigns each item to source wells.
def calculate_source_wells(volumes_df, triplicate, source_start_well, max_volume_per_well):
    if triplicate:
        repetition = 3
    else:
        repetition = 1

    # Initialize variables
    rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    columns = list(range(1, 25))
    well_coordinates = [f"{row}{col}" for row in rows for col in columns]

    # Find starting well index
    start_row = source_start_well[0]
    start_col = int(source_start_well[1:])
    well_index = well_coordinates.index(f"{start_row}{start_col}")

    source_wells = []

    for substrate in volumes_df.columns:
        # Calculate total volume needed for the substrate
        total_volume = repetition * volumes_df[substrate].sum()

        # Calculate the number of source wells needed
        num_source_wells = math.ceil(total_volume / max_volume_per_well)

        # Check if there are enough remaining wells
        if well_index + num_source_wells > len(well_coordinates):
            raise ValueError(f"Not enough wells available including and after the following item: {substrate}. Consider using a new source plate")

        remaining_volume = total_volume

        for _ in range(num_source_wells):
            if remaining_volume >= max_volume_per_well:
                source_wells.append({
                    "Item": substrate,
                    "Source Well": well_coordinates[well_index],
                    "Volume": max_volume_per_well
                })
                remaining_volume -= max_volume_per_well
            else:
                source_wells.append({
                    "Item": substrate,
                    "Source Well": well_coordinates[well_index],
                    "Volume": remaining_volume
                })
                remaining_volume = 0
            well_index += 1

    source_wells_df = pd.DataFrame(source_wells)

    return source_wells_df

# This function generates a table instruction for source well pipetting.
def source_pipetting_instructions(source_wells_df, fixed_parts, Master_Mix_for_fixed_parts=True, buffer_volume=30000):
    """
    Produces a pipetting instruction for your source well.

    Parameters
    ----------
    source_wells_df: DataFrame
        DataFrame that includes source wells with volumes.

    fixed_parts: dict
        Dictionary of fixed parts with their corresponding proportions.

    Master_Mix_for_fixed_parts: bool
        If True, delete rows corresponding to items in fixed_parts.

    buffer_volume: int
        Volume of buffer to be added to each source well.

    Returns
    -------
    source_wells_instruction: DataFrame
        Updated DataFrame with buffer volume, total volume columns and fixed parts removed if applicable.
    """

    # Add buffer volume column
    source_wells_df['Buffer Volume'] = buffer_volume

    # Add total volume column
    source_wells_df['Total Volume'] = source_wells_df['Volume'] + source_wells_df['Buffer Volume']

    # Remove fixed parts if Master_Mix_for_fixed_parts is True
    if Master_Mix_for_fixed_parts:
        fixed_parts_items = set(fixed_parts.keys())
        source_wells_df = source_wells_df[~source_wells_df['Item'].isin(fixed_parts_items)]
        source_wells_df = source_wells_df.reset_index(drop=True)

    return source_wells_df
