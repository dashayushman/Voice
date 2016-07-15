from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

data = pd.read_csv('iris.data')
plt.figure()
parallel_coordinates(data, 'Name')
plt.show()