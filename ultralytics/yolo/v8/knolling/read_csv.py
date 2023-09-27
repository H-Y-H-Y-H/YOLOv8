import pandas as pd

path = '../../../../../Knolling_bot_2/models/919_grasp/'

data = pd.read_csv(path + 'results.csv')

print(data)