from Libraries.lib_excel import generate_acc_transactions, generate_car_transactions

import os
import pandas as pd
import numpy as np
from tabulate import tabulate

####################################################################################################

# Function to ...
def function_name(arguments):
    return

####################################################################################################

# Function to get data from all sources
def test001():
    while True:
        category_id_new = input(">>> CATEGORY_ID: \n(for displaying the list of categories and IDs press ENTER)\n")
        if category_id_new == "":
            print("This is the list of categories and IDs:")
            # Display list of categories and IDs (dataframe df_categories)
            print('print(tabulate(df_categories, headers=''keys'', tablefmt=''psql'', showindex=False))')
            continue
        elif category_id_new != "a":
            print(f"\n!! category_id {category_id_new} not in database")
            continue
        else:
            break