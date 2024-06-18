# Useful Things I learnt

## 1. inplace = True/False in df.drop

- df.drop([column_name], axis=1, inplace=False) -> default

  - returns the df hence,
    should be
    df = df.drop([column_name], axis=1, inplace=False)

  Else, if inplace=True, it is renamed in its place
  df.drop([column_name], axis=1, inplace=False) should update the df!
