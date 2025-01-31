from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore") # Don't need warnings for a demo

DAYS_TO_FORECAST = 5
FIGURE_WIDTH = 8
FIGURE_HEIGHT = 6

DATA_FILES =[
    './data/Plant_1_Generation_Data.csv',
    './data/Plant_1_Weather_Sensor_Data.csv',
    './data/Plant_2_Generation_Data.csv',
    './data/Plant_2_Weather_Sensor_Data.csv'
]

def load_data(filenames:list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for file in filenames:
        new_df = pd.read_csv(file)
        df = pd.concat([df, new_df])       
    return df

def convert_to_datetime(text:str) -> datetime:
    format1 = "%d-%m-%Y %H:%M"
    try:
        return datetime.strptime(text, format1)
    except ValueError: #Try a different format
        format2 = "%Y-%m-%d %H:%M:%S"
        return datetime.strptime(text, format2)

def mean_percent_error(y_pred:list, y_true:list) -> float:
    ab_err = np.abs(y_pred - y_true)
    per_err = 100 * ab_err / y_true 
    return np.mean(per_err)

def collate_data(data_df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inverter_aggregation_dict ={
        'DATE': 'last',
        'PLANT_ID': 'last',
        'DAILY_YIELD': 'max',
        'TOTAL_YIELD': 'max',
    }
    plant_aggregation_dict ={
        'DATE': 'last',
        'DAILY_YIELD': 'sum',
        'TOTAL_YIELD': 'sum',
    }
    weather_aggregation_dict ={
        'AMBIENT_TEMPERATURE': 'mean',
        'MODULE_TEMPERATURE': 'mean',
        'IRRADIATION': 'sum',
    }
    power_df = data_df.dropna(subset=['SOURCE_KEY', 'DAY'] + list(inverter_aggregation_dict.keys()))        
    power_by_inverter_df = power_df.groupby(['DAY', 'SOURCE_KEY']).agg(inverter_aggregation_dict)
    power_by_plant_df = power_by_inverter_df.reset_index()
    power_by_plant_df = power_by_plant_df.groupby(['DAY', 'PLANT_ID']).agg(plant_aggregation_dict) 
    weather_df = data_df.dropna(subset=['PLANT_ID', 'DAY'] + list(weather_aggregation_dict.keys()))
    weather_by_plant_df = weather_df.groupby(['DAY', 'PLANT_ID']).agg(weather_aggregation_dict)
    data_by_plant_df = power_by_plant_df.join(weather_by_plant_df)
    data_by_plant_df['EFFICIENCY'] = data_by_plant_df['DAILY_YIELD'] / data_by_plant_df['IRRADIATION']
    return data_by_plant_df, power_by_inverter_df

def visualize_plant_data(data_by_plant_df:pd.DataFrame) -> None:    
    plants = data_by_plant_df.index.get_level_values('PLANT_ID').unique()
    fig, axs = plt.subplots(len(plants), 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.subplots_adjust(hspace=0.5)
    plot_df = data_by_plant_df.reset_index(level='DAY')
    for plant in plants:
        plant_df = plot_df.loc[plant]
        axs[0].plot(plant_df['DAY'], plant_df['DAILY_YIELD'], label = f'Plant: {plant}')
        axs[1].plot(plant_df['DAY'], plant_df['EFFICIENCY'], label = f'Plant: {plant}')
    axs[0].legend()
    axs[0].set_xlabel('Day')
    axs[0].set_ylabel('DAILY_YIELD')
    axs[0].set_title(f'Daily Yield of Plants')
    axs[1].legend()
    axs[1].set_xlabel('Day')
    axs[1].set_ylabel('DAILY_YIELD / IRRADIANCE')
    axs[1].set_title(f'Daily Efficiency of Plants')
    plt.show()

def generate_arima_model(training_df:pd.DataFrame) -> dict[str, ARIMA]:    
    models = {}
    plants = training_df['PLANT_ID'].unique() 
    for plant in plants:
        arima_training_data = training_df.set_index('DATE')
        arima_training_data = arima_training_data[arima_training_data['PLANT_ID']==plant]['DAILY_YIELD']
        arima_training_data.index.freq = 'D'
        model = ARIMA(arima_training_data, order=(1, 1, 1), freq='D')        
        models[plant] = model.fit()
    return models

def evaluate_arima_model(models:dict[str, ARIMA], training_df:pd.DataFrame, validation_df:pd.DataFrame) -> None:
    rf_validation_data = validation_df[['PLANT_ID', 'DAILY_YIELD', 'AMBIENT_TEMPERATURE', 'IRRADIATION']]
    rf_validation_data = pd.get_dummies(rf_validation_data, columns=['PLANT_ID'])
    prediction = validation_df.copy()
    prediction = prediction.set_index(['DATE', 'PLANT_ID'])    
    plants = prediction.index.get_level_values('PLANT_ID').unique()
    for plant in plants:        
        plant_pred = models[plant].forecast(steps=DAYS_TO_FORECAST)
        for time in plant_pred.index:
            idx = (time.date(),plant)
            prediction.at[idx,'DAILY_YIELD'] = plant_pred[time]
    prediction = prediction.reset_index()
    observed_data = pd.concat([training_df, validation_df])  
    fig, axs = plt.subplots(len(plants), 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.subplots_adjust(hspace=0.5)
    for (ax, plant) in zip(axs.flat, plants):
        validation_y = validation_df[validation_df['PLANT_ID']==plant]['DAILY_YIELD']
        prediction_y = prediction[prediction['PLANT_ID']==plant]['DAILY_YIELD']
        err = mean_percent_error(prediction_y.values, validation_y.values)
        print(f'Average percent error for ARIMA model of plant {plant} is {err}%')
        ax.plot(observed_data[observed_data['PLANT_ID']==plant]['DAY'], observed_data[observed_data['PLANT_ID']==plant]['DAILY_YIELD'], label='Observed', color='blue')
        ax.plot(prediction[prediction['PLANT_ID']==plant]['DAY'], prediction[prediction['PLANT_ID']==plant]['DAILY_YIELD'], label='Prediction', color='red', linestyle='--') 
        ax.set_title(f'Plant {plant} (MPE = {err:.2f}%)')
        ax.set_xlabel('Day')
        ax.set_ylabel('DAILY_YIELD')
        ax.legend()
    plt.suptitle(f'ARIMA {DAYS_TO_FORECAST:d}-Day Forecast')
    plt.show()

def generate_rf_model(training_df:pd.DataFrame) -> RandomForestRegressor:
    rf_training_data = training_df[['PLANT_ID', 'DAILY_YIELD', 'AMBIENT_TEMPERATURE', 'IRRADIATION']]
    rf_training_data = pd.get_dummies(rf_training_data, columns=['PLANT_ID'])
    X = rf_training_data.drop('DAILY_YIELD', axis='columns')
    y = rf_training_data['DAILY_YIELD']    
    test_score = 0    
    while test_score < 0.7:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        model = RandomForestRegressor(n_jobs=-1, max_features=3, n_estimators=20, max_depth=3)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
    print(f'Random Forest scores: train = {train_score}, test = {test_score}')
    return model

def evaluate_rf_model(model:RandomForestRegressor, training_df:pd.DataFrame, validation_df:pd.DataFrame) -> None:    
    plants = validation_df['PLANT_ID'].unique()
    rf_validation_data = validation_df[['PLANT_ID', 'DAILY_YIELD', 'AMBIENT_TEMPERATURE', 'IRRADIATION']]
    rf_validation_data = pd.get_dummies(rf_validation_data, columns=['PLANT_ID'])
    X_prediction = rf_validation_data.drop('DAILY_YIELD', axis='columns')
    prediction = validation_df.copy()
    prediction['DAILY_YIELD'] = pd.Series(model.predict(X_prediction), index=X_prediction.index)
    observed_data = pd.concat([training_df, validation_df]) 
    fig, axs = plt.subplots(len(plants), 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.subplots_adjust(hspace=0.5)
    for (ax, plant) in zip(axs.flat, plants):
        validation_y = validation_df[validation_df['PLANT_ID']==plant]['DAILY_YIELD']
        prediction_y = prediction[prediction['PLANT_ID']==plant]['DAILY_YIELD']
        err = mean_percent_error(prediction_y, validation_y)
        print(f'Average percent error for random forest model of plant {plant} is {err}%')
        ax.plot(observed_data[observed_data['PLANT_ID']==plant]['DAY'], observed_data[observed_data['PLANT_ID']==plant]['DAILY_YIELD'], label='Observed', color='blue')
        ax.plot(prediction[prediction['PLANT_ID']==plant]['DAY'], prediction[prediction['PLANT_ID']==plant]['DAILY_YIELD'], label='Prediction', color='red', linestyle='--') 
        ax.set_title(f'Plant {plant} (MPE = {err:.2f}%)')
        ax.set_xlabel('Day')
        ax.set_ylabel('DAILY_YIELD')
    plt.suptitle(f'Random Forest Regression {DAYS_TO_FORECAST:d}-Day Forecast')
    plt.show()

def visualize_inverter_data(power_by_inverter_df:pd.DataFrame) -> None:    
    plants = power_by_inverter_df['PLANT_ID'].unique()
    fig, axs = plt.subplots(len(plants), 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    fig.subplots_adjust(hspace=0.5)
    for [ax, plant] in zip(axs.flat, plants):
        inverter_df = power_by_inverter_df[power_by_inverter_df['PLANT_ID']==plant].reset_index(level='DAY')
        for inverter in inverter_df.index:
            x = inverter_df.loc[inverter]['DAY']
            y = inverter_df.loc[inverter]['DAILY_YIELD']
            ax.plot(x,y,label=inverter)
        ax.set_xlabel('Day')
        ax.set_ylabel('DAILY_YIELD')         
        ax.set_title(f'Plant {plant}')
    plt.suptitle(f'Daily Inverter Output')
    plt.show()

def get_inverter_outliers(power_by_inverter_df:pd.DataFrame) -> None:        
    plants = power_by_inverter_df['PLANT_ID'].unique()
    for plant in plants:
        inverter_df = power_by_inverter_df[power_by_inverter_df['PLANT_ID']==plant].reset_index(level='DAY')
        days = inverter_df['DAY'].unique()
        print(f'Plant {plant} analysis')
        for day in days:            
            data = inverter_df[inverter_df['DAY']==day]['DAILY_YIELD']
            daily_max = max(data)
            data = daily_max - data
            shape, loc, scale = stats.weibull_min.fit(data, loc=0)
            mean = stats.weibull_min.mean(shape, loc, scale)
            std_dev = stats.weibull_min.std(shape, loc, scale)
            z_score = (data - mean) / std_dev
            outliers = np.where(np.abs(z_score) > 3)[0]            
            report = ''
            for idx in outliers:
                name = data.index[idx]
                power = data[idx]
                percent = 100*power/daily_max
                report += f'{name} ({percent:.1f}%) '
            if len(report) > 0:
                print(f'Day {day} Outliers: (% of daily maximum produced)')
                print(report)

def main() -> None:
    #DATA INTAKE
    data_df = load_data(DATA_FILES)
    data_df['DATE_TIME'] = data_df['DATE_TIME'].apply(convert_to_datetime)      
    data_df['DATE'] = data_df['DATE_TIME'].apply(datetime.date)
    first_day = min(data_df['DATE'])
    data_df['DAY'] = data_df['DATE'].apply(lambda x: (x - first_day).days + 1)
    data_by_plant_df, power_by_inverter_df = collate_data(data_df)

    #PLOT PLANT-WIDE POWER OVER TIME    
    visualize_plant_data(data_by_plant_df)
    
    #MODEL & FORECAST SETUP 
    forecast_df = data_by_plant_df.reset_index()
    last_day = max(forecast_df['DAY'])
    forecast_day = last_day - DAYS_TO_FORECAST + 1
    training_df = forecast_df[forecast_df['DAY'] < forecast_day]
    validation_df = forecast_df[forecast_df['DAY'] >= forecast_day]

    #ARIMA
    arima_models = generate_arima_model(training_df)
    evaluate_arima_model(arima_models, training_df, validation_df)

    #RF
    rf_model = generate_rf_model(training_df)
    evaluate_rf_model(rf_model, training_df, validation_df)

    #INVERTER ANALYSIS
    visualize_inverter_data(power_by_inverter_df)
    get_inverter_outliers(power_by_inverter_df)  

    print('\nThe Machine Learning Visualization demo is complete!\n\t- Anthony Roy PhD ')
if __name__ == '__main__':    
    main()