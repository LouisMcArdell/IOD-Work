def quick_eda(dataframe):
    '''
    A basic function to output some simple EDA commands.
    prints;
        head
        columns
        shape
        data types
        missing values
        summary statistics
        correlations
    '''
    print('--- Head ---')
    display(dataframe.head())
    print('\n')
    
    print('--- Columns ---')
    print(dataframe.columns)
    print('\n')

    print('--- Shape ---')
    print(f'Number of Rows: {dataframe.shape[0]}')
    print(f'Number of Columns: {dataframe.shape[1]}')
    print('\n')

    print('--- Data Types ---')
    print(dataframe.dtypes)
    print('\n')

    print('--- Missing Values ---')
    print(dataframe.isnull().sum())
    print('\n')

    missing_values = dataframe.isnull().sum()
    percentage_missing = (missing_values / len(dataframe.index) * 100).round(2)
    sorted_percentage_missing = percentage_missing.sort_values(ascending=False).head(18)
    print(sorted_percentage_missing)

    print('--- Summary Statistics ---')
    display(dataframe.describe())
    print('\n')

    print('--- Correlations ---')
    display(dataframe.corr(numeric_only=True))

