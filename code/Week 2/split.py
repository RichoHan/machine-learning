import sframe

sf = sframe.SFrame.read_csv('data/home_data.csv')

train_data, test_data = sf.random_split(0.8, seed=0)

df_train = train_data.to_dataframe()
df_test = test_data.to_dataframe()

df_train.to_csv('data/home_train_data.csv')
df_test.to_csv('data/home_test_data.csv')
