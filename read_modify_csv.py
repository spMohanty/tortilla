import pandas as pd
from utils import query_yes_no

def modify_csv(EXP):
    response = query_yes_no("Do you want to comment one of these experiments?",default="no")

    choice = ["General_impression", "Suggested_changes"]

    if response:
        while True:
            try:
                name = input("Please enter the name of the experiment: ")
                variable = input("Please enter the name of the variable (General_impression or Suggested_changes): ")
                if variable not in choice:
                    print("Error with the arguments, try again.\n")
                    continue
                comment = input("Please enter the comment: ")
            except:
                name = raw_input("Please enter the name of the experiment: ")
                variable = raw_input("Please enter the name of the variable (General_impression or Suggested_changes): ")
                if variable not in choice:
                    print("Error with the arguments, try again.\n")
                    continue
                comment = raw_input("Please enter the comment: ")

            try:
                EXP[name].loc['Comments', variable] = comment
                print(EXP)
                NEW_EXP = modify_csv(EXP)
                return NEW_EXP
            except:
                print("Error with the arguments, try again.\n")
    else :
        return EXP

if __name__ == "__main__":
    EXP = pd.read_csv('Experiments.csv', sep=';', index_col=['Type','Variables'])
    print(EXP)
    NEW_EXP = modify_csv(EXP)
    NEW_EXP.to_csv('Experiments.csv', sep=';')
