# %% [markdown]
# ## Project Overview

# %% [markdown]
# Global warming, Climate change, and Human health are getting impacted due to excessive agri-food emissions. Hence, the predictive analysis of average temperature increase that is caused by the CO2 emissions from agri-food activities is important for policymakers and researchers to develop strategies for sustainable agricultural practices. This project explores historical data on agri-food CO2 emissions and it's impact on the increase of temperature in various countries around the world for a time span of 30 years (1990–2020). Since there is a need for predicting emission from the agri-food sector and corresponding temperature increase, this project explores this area by implementing the three predictive models Linear Regression, Decision Trees, Random Forests. Exploratory data analysis (EDA) helps to understand the descriptive statistics, and data visualizations on agri-food activities, emissions, temperature rise, and their relationships. The three predictive models are trained and measured with metrics like MSE, RMSE, MAE, and R-squared. The Linear Regression model emerged as the best model with the highest predictive accuracy, with the lowest RMSE, MAE and highest R2-score for CO2 emissions. The project concludes that Linear Regression can serve as a robust tool in predicting tempereture increase from CO2 emissions from agri-food activities and helps the policymakers, government bodies, and sustainable environment by providing useful insights and strategies to reduce the environmental impact of agriculture.
# 
# **This Project aims to:**
# Understand the relationship between Temperature rise and CO2 emission derriving from the agri-food sector activities.
# Extensively investigate the relationship amongst an extensive list of agri-food processes CO2 mmisions and average temperature rise.
# Identify any potential correlations between the agri-food sector mmissions and average temperature increase.

# %% [markdown]
# **Import Packages**

# %%
#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.experimental import enable_iterative_imputer # We need .experimental, .impute, .linear_model for data imputation using the MICE framework. 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler 



# %% [markdown]
# ## Load Data

# %%
# Load the data. 
df = pd.read_csv("co2_emissions_from_agri.csv")

# %% [markdown]
# **Data dictionary**
# 
# 
# 

# %% [markdown]
# ## Cleaning Data

# %%
# Check the first few rows of the dataset
df.head()

# %%
# Display basic information about the dataset
df.info()

# %%
# Count missing values in each column
df.isnull().sum()  

# %%
# Check for duplicates
df.duplicated().sum()

# %%
#Check how many rows and how many columns
df.shape

# %%
# Standardize column names
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# %%
# Check the first few rows of the dataset
df.head()

# %%
df.info()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
# Checking for outliers (basic statistics)
df.describe()

# %%
sns.set_style("darkgrid") #set the style of your preference
fig, ax =plt.subplots(figsize = (16,8)) #set figure and axes
sns.lineplot(data = df, x = 'YEAR', y = 'AVERAGE_TEMPERATURE_°C', ax = ax) #plot a lineplot with Year in the x-axis and the average temperature in the y-axis
fig.suptitle('Average Temperature time series') #choose a title

# %% [markdown]
# Since 1990, the average land temperature line, compared to pre-industrial times, showcases an upward trend, hitting an all time high of approximately 1.5°C in 2020. 

# %%
#If the values are not standardized, we won't be able to plot the lines in the same graph, as the scales will be totally different. 
from sklearn.preprocessing import MinMaxScaler # Import min-max scaler. This will transform all the variable to take values between 0 and 1.

scaler = MinMaxScaler() #let's instantiate it.

temp_emission = df.groupby("YEAR").agg({"AVERAGE_TEMPERATURE_°C": "mean", "TOTAL_EMISSION": "mean", 'URBAN_POPULATION': 'mean'}) # let's calculate 
                                                                                                                                #the mean emission, 
                                                                                                                                #population and temperature per year
norm_emission= scaler.fit_transform(temp_emission) #Here we transform the mean values using the scaler.

temp_df = pd.DataFrame(norm_emission, columns = ['Standardized Avg Temperature','Standardized Mean CO2 emission', 'Standardized Mean Urban Population']) #because the scaler outputs 
                                                                                                                                                    ##Nparrays, let's create a df.
temp_df.index = [i for i in range(1990, 2021)] # the data were grouped by year and hence, we can use the year as index. The starting year is 1990 and the end year 2020

# %%
fig, ax = plt.subplots(figsize = (10,6)) # Let's set our figure and axes

g = sns.lineplot(temp_df, ax = ax) #Use seaborn because its looking cool
fig.suptitle('Normalized CO2 Emission, Temperature, urban population')
plt.plot() # and plot

# %% [markdown]
# The graph above shows that CO2 emission, mean urban population and average temperature go hand in hand. All three lines show a strong upward trend.

# %%
from matplotlib.ticker import FuncFormatter # We gonna use this to convert a big number to millions

temp = df.iloc[:,  1:].groupby('YEAR').agg({'AVERAGE_TEMPERATURE_°C':'mean', 'TOTAL_EMISSION':'sum'} ) #Let's group by year and calculate mean annual temperature 
                                                                                                       ## and total annual emissions for all countries cummulatively.
fig, ax = plt.subplots(figsize = (12, 6))
g = sns.scatterplot(data = temp, x = 'TOTAL_EMISSION', y = 'AVERAGE_TEMPERATURE_°C', ax = ax, hue ='YEAR', palette = "mako") 

def millions_formatter(x, pos):  # Let's define a function that takes a number divides it by 1000000 and returns it in millions.
    return '{:,.0f}M'.format(x / 1000000) 

# Apply the formatter to the x-axis ticks
ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))

fig.suptitle('Global Average Annual Temperature vs Total Annual CO2 emissions scatterplot')
ax.set_xlabel('Total annual CO2 emissions in million tons')
plt.show()

# %% [markdown]
# Total emissions and temperature are gradually increasing over time which indicates a potential 
# correlation between them. Both CO2 emissions and temperature continue to rise sharply i 
# recent years, indicating ongoing environmental challenges with respect to global warming a d
# greenhouse gas emissions.

# %%
temp = df.iloc[:, 2: -7].copy() # Do not include the first and second column as it is area and year
delete_list = ['FORESTLAND', 'NET_FOREST_CONVERSION', 'ON-FARM_ELECTRICITY_USE'] #rule out all non-relevant features
temp = temp.drop(delete_list, axis = 1)
means = temp.mean()#calculate the mean anual values for each feature
means.sort_values(ascending = False, inplace = True) #sort them in descending order
cols = means.index #the column names are the indices

# %%
fig, ax = plt.subplots(figsize=(16, 12))
sns.barplot(x=means, y=cols, ax=ax) #wrap x and y parameters

ax.set_ylabel("Activity")  # Set the y-axis label
ax.set_xlabel("Mean CO2 emmissions (in tons)")     # Set the x-axis label
fig.suptitle('Mean CO2 emmissions (in tons) by Agri-food activity')

plt.xticks(rotation=45)      # Rotate x-axis labels for better visibility
plt.tight_layout()           # Adjust layout to prevent labels from getting cut off
plt.show()

# %% [markdown]
# The above graph shows that IPPU is by far the acctivity with the most CO2 emissions, followed by emissions coming from waste disposal and agrifood products consumption. Thereafter, other industrial and cultivation activities follow, with fire emissions coming towards the end

# %%
df.groupby('AREA')['TOTAL_EMISSION'].sum().sort_values(ascending=False).head(20).plot.bar();

# %% [markdown]
# The graph above shows that China has the highest CO2 emission, followed by Brazil,Indonesia and USA.

# %%
df.groupby('AREA')['TOTAL_EMISSION'].sum().sort_values(ascending=True).head(20).plot.bar();

# %% [markdown]
# This graph shows that Russia has the lowest CO2 emission.

# %%
# Select relevant columns
selected_columns = ['AREA', 'TOTAL_EMISSION', 'AGRIFOOD_SYSTEMS_WASTE_DISPOSAL', 'FOOD_HOUSEHOLD_CONSUMPTION',
                    'FOREST_FIRES', 'FIRES_IN_HUMID_TROPICAL_FORESTS', 'SAVANNA_FIRES',
                    'IPPU', 'FOOD_PROCESSING', 'FOOD_TRANSPORT', 'FOOD_PACKAGING', 'PESTICIDES_MANUFACTURING',
                    'FERTILIZERS_MANUFACTURING', 'FOOD_RETAIL',
                    'RICE_CULTIVATION', 'MANURE_LEFT_ON_PASTURE', 'DRAINED_ORGANIC_SOILS_(CO2)',
                    'CROP_RESIDUES', 'MANURE_MANAGEMENT', 'MANURE_APPLIED_TO_SOILS', 'AVERAGE_TEMPERATURE_°C', 'YEAR']

# Create a simplified DataFrame with selected columns
temp2 = df[selected_columns].copy()

# Calculate the categories total values
emission_sources = ['FOREST_FIRES', 'FIRES_IN_HUMID_TROPICAL_FORESTS', 'SAVANNA_FIRES',
                    'IPPU', 'FOOD_PROCESSING', 'FOOD_TRANSPORT', 'FOOD_PACKAGING', 'PESTICIDES_MANUFACTURING',
                    'FERTILIZERS_MANUFACTURING', 'FOOD_RETAIL',
                    'RICE_CULTIVATION', 'MANURE_LEFT_ON_PASTURE', 'DRAINED_ORGANIC_SOILS_(CO2)',
                    'CROP_RESIDUES', 'MANURE_MANAGEMENT', 'MANURE_APPLIED_TO_SOILS']

temp2['TOTAL_FIRE_EMISSIONS'] = temp2[emission_sources[:3]].sum(axis=1)
temp2['TOTAL_INDUSTRIAL_EMISSIONS'] = temp2[emission_sources[3:12]].sum(axis=1)
temp2['TOTAL_CULTIVATION_EMISSIONS'] = temp2[emission_sources[12:]].sum(axis=1)


# Calculate mean values for all emission sources
means = temp2[['TOTAL_FIRE_EMISSIONS', 'TOTAL_INDUSTRIAL_EMISSIONS', 'TOTAL_CULTIVATION_EMISSIONS',
               'AGRIFOOD_SYSTEMS_WASTE_DISPOSAL', 'FOOD_HOUSEHOLD_CONSUMPTION']].mean()

means.rename({'TOTAL_FIRE_EMISSIONS' : 'mean_fire_emissions', 'TOTAL_INDUSTRIAL_EMISSIONS' : 'mean_industrial_emissions', 
                       'TOTAL_CULTIVATION_EMISSIONS' : 'mean_cultivation_emissions', 
                       'AGRIFOOD_SYSTEMS_WASTE_DISPOSAL' :  'mean Agrifood Systems Waste Disposal', 
                       'FOOD_HOUSEHOLD_CONSUMPTION' : 'mean Food Household Consumption'})


# Sort columns based on mean values in descending order
means.sort_values(ascending=False, inplace = True)

cols = means.index

# %%
fig, ax = plt.subplots(figsize=(10, 8)) 
sns.barplot(x=means, y=cols, ax=ax) #wrap x and y parameters

ax.set_ylabel("Activity")  # Set the y-axis label
ax.set_xlabel("Mean CO2 emmissions (in tons)")     # Set the x-axis label
fig.suptitle('Mean CO2 emmissions (in tons) by Agri-food activity')
      
plt.tight_layout()           # Adjust layout to prevent labels from getting cut off
plt.show() #

# %% [markdown]
# The bar chart above shows that industrial activities emit the biggest proportion of CO2, followed by cultivation emissions (as opposed to waste disposal emissions), which in turn are followed by waste disposal emissions, consumption emissions, and fire emissions.

# %%
# Let's plot the distribution plots using subplot2grid to adjust the position and seaborn because its cool.

fig, ax = plt.subplots(figsize = (32, 16))
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

sns.histplot(data = temp2, x='TOTAL_INDUSTRIAL_EMISSIONS', ax = ax1, bins = 50)
sns.histplot(data = temp2, x='TOTAL_CULTIVATION_EMISSIONS', ax = ax2, bins = 50)
sns.histplot(data = temp2, x='TOTAL_FIRE_EMISSIONS', ax = ax3, bins = 50)
sns.histplot(data = temp2, x='AGRIFOOD_SYSTEMS_WASTE_DISPOSAL', ax = ax4, bins = 50)
sns.histplot(data = temp2, x='FOOD_HOUSEHOLD_CONSUMPTION', ax = ax5, bins = 50)

fig.suptitle('Distribution Graphs', fontsize = 18)

plt.show()

# %% [markdown]
# The data are negatively skewed for all emission categories. This means that the histograms have a long right tail. The underlying reason for this behavior might be that the must majority of countries have a small or moderate agrifood sector and hence the emissions are much lower in comparison to agrifood and industrial giants such as China, the USA or Brazil.

# %%
#choose the relevant variables
variables = ['TOTAL_EMISSION','TOTAL_FIRE_EMISSIONS',
       'TOTAL_INDUSTRIAL_EMISSIONS', 'TOTAL_CULTIVATION_EMISSIONS',
       'AGRIFOOD_SYSTEMS_WASTE_DISPOSAL',
       'FOOD_HOUSEHOLD_CONSUMPTION', 'AVERAGE_TEMPERATURE_°C', 'YEAR' ]

#add them to temp3
temp3 = temp2[variables].copy()

sns.heatmap(data = temp3.corr(), annot = True)

# %%
# Calculate and add Global Temperature mean per year to temp4
temp4 = temp3.copy()
temp4 = temp4.groupby('YEAR').transform('mean')

# %%
temp4['YEAR'] = temp3['YEAR']
sns.pairplot(data = temp4, hue = 'YEAR')
fig.suptitle('Critical variables aggregated by year scatterplot', fontsize = 18)

# %%
sns.heatmap(data = temp4.corr(), annot = True)

# %% [markdown]
# The positive relationship amongst most predictor and Global Temperature is demonstrated in the heatmap above.

# %% [markdown]
# The average temperature increase, along with urban population and total emissions showcase an upward trend.
# Industrial processes emit the most CO2 in the agri-food sector, while fire emissions the least.
# China, India, Brazil and the USA emit the most CO2 from the agri-food sector.
# Temperature rise seems to be strongly related to Year and Area.
# Fire emissions do not seem to be related to temperature rise.
# All agri-food sector processes emissions seem to be strongly, positively related to the increase in the average global temperature.

# %% [markdown]
# **Model Evaluation Metrics**

# %% [markdown]
# Model evaluation metrics such as MSE, RMSE, MAE, and R2 are used in this project to measure the performance of predictive models. These metrics show how well a model predicts average temperature rise which assists in selection of the most accurate and reliable model.

# %% [markdown]
# **Handling Missing Values**

# %% [markdown]
# Before we can run the models we need to deal with the null values present in our dataset.There are quite a few null values that we need to deal with. It is important to understand whether the null values are null because data was not collected or because there is some meaning into it. We will need to check each variable seperately in order to make sense. Let's have a look at the area,maybe it will give us some insight.

# %%
#Let's start with Savanna fires. 
##I will make a new df series  'bool_series', that holds the boolean values of whether the Savanna fires column contains null values.
df['bool_series'] = pd.isnull(df["SAVANNA_FIRES"])

# %%
# now using 'bool_series', I will check which areas have null values in the column 'Savanna fires'.
df.loc[df['bool_series'], 'AREA'].unique()

# %% [markdown]
# The data shows that Holy See is the only area that has no records of Savanna values. This makes sence since it is not possible that such fire events occur in the city of Vatican. We will also need to check what other null values this Area has. It is a special case of an area since many features such as Forest fires, Savanna fires, manure, and tropical forest fires or farms do not exist.

# %%
# Let's see
null = df[df['AREA'] == 'Holy See'].isnull()

# %%
null.sum()

# %% [markdown]
# The data shows that those features contain null values because in reality they are non existent in the Holy see therefore we will fill all the null values with 0.

# %%
df.loc[df['AREA'] == 'Holy See'] = df[df['AREA'] == 'Holy See'].fillna(0)

# %%
#Done.
df.loc[df['AREA'] == 'Holy See']

# %% [markdown]
# Let's have a look at Forest fires

# %%
df['bool_series'] = pd.isnull(df["FOREST_FIRES"])

# %%
df.loc[df['bool_series'], 'AREA'].unique()

# %% [markdown]
# The data shows that Monaco and San Marino have null values for Forest fires. after investigating these areas we conclude that is reasonable that data has null values because there are no forest in those two areas.

# %%
#Here we locate the forest column of df only for the Areas of Monaco and San Marino and replace the null values with 0.
df.loc[(df['AREA'] == 'Monaco')|(df['AREA'] == 'San Marino'), ['FOREST_FIRES']] = df.loc[(df['AREA'] == 'Monaco') | (df['AREA']=='San Marino'), ['FOREST_FIRES']].fillna(0)

# %%
#Done
df.loc[(df['AREA'] == 'Monaco')|(df['AREA'] == 'San Marino')]

# %% [markdown]
# Let's check Fires in humid tropical forests.

# %%
df['bool_series'] = pd.isnull(df['FIRES_IN_HUMID_TROPICAL_FORESTS'])

# %%
df['bool_series'].sum()

# %%
df.loc[df['bool_series'], 'AREA'].unique()

# %% [markdown]
# Those 4 areas do not have humid tropical forest and hence the null values will be replaced to 0.

# %%
#Done.
df.loc[df['bool_series'], 'FIRES_IN_HUMID_TROPICAL_FORESTS'] = df.loc[df['bool_series'], 'FIRES_IN_HUMID_TROPICAL_FORESTS'].fillna(0)

# %% [markdown]
# We have identified all the missing values that can be filled reasonably(MNAR - Missing Data Not at Random) and have corrected them. The rest of the variables include many missing values, most probably because there are simply no such records present in the dataset at random (MAR - Missing Data at Random). We will use linear regression under the MICE (Multiple Imputation by Chained Equation) framework to fill the rest of the missing values.

# %%
#first we select all variables from df and write a list of the column names in the object num_features.
num_features = [col for col in df.columns]

# %%
#Area should be removed because in linear regression only continuous variables can be included.
num_features.remove('AREA')

# %%
#let's perform MICE. 
#call LinearRegression()
lr = LinearRegression()
#call the imputer and specify the estimator as lr, define the missing values as np.nan, and use a max_iteration stopping criterion.10 cycles should be enough.
##set random_state = 0 for reproducibility.also set the imputation_order as 'roman' which simply takes the variables from left to right. Verbose = 2, showing one line per cycle.
imp = IterativeImputer(estimator=lr,missing_values=np.nan,  max_iter=10, verbose=2, imputation_order='roman',random_state=0)
X=imp.fit_transform(df.iloc[:, 1:])

# %%
#Now let's create df2 which includes the imputed values
df2 = pd.DataFrame(X, columns = num_features)

# %%
#No null values included
df2.isnull().sum()

# %%
X = df2.iloc[:,0:-1]
X

# %% [markdown]
# **Perform train-test splits**

# %%
y = df2.iloc[:,-2]
y

# %%
selector=SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X,y)

# Get the selected feature indices
feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X.columns[feature_indices]

# Create a new DataFrame with only the selected features
selected_data = df[selected_features]
# Print the selected features
print(selected_features)

# %%
X=df2[['YEAR', 'SAVANNA_FIRES', 'FOOD_HOUSEHOLD_CONSUMPTION', 'FOOD_RETAIL',
       'FOOD_PACKAGING', 'FOOD_PROCESSING', 'IPPU', 'FIRES_IN_ORGANIC_SOILS',
       'FIRES_IN_HUMID_TROPICAL_FORESTS', 'AVERAGE_TEMPERATURE_°C']]

# %%
#scaling using standard scaler
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_ss=ss.fit_transform(X)

# %%
#Performing train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_ss,y,test_size=0.2,random_state=0)

# %%
#linear regression
mr=LinearRegression()
mr.fit(X_train,y_train)
y_pred=mr.predict(X_test)
print("mean absolute error:",mean_absolute_error(y_test,y_pred))
print("mean squared error:",mean_squared_error(y_test,y_pred))
print("root mean squared error:",np.sqrt(mean_squared_error(y_test,y_pred)))
print("r2-score:",r2_score(y_test,y_pred))

# %%
#random forest regrssor
rs=RandomForestRegressor()
rs.fit(X_train,y_train)
y_pred1=rs.predict(X_test)
y_pred1
print("mean absolute error:",mean_absolute_error(y_test,y_pred1))
print("mean squared error:",mean_squared_error(y_test,y_pred1))
print("root mean squared error:",np.sqrt(mean_squared_error(y_test,y_pred1)))
print("R2 score:",r2_score(y_test,y_pred1)) 

# %%
#DecisionTree regressor
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
y_pred7=dt.predict(X_test)
y_pred7
print("mean absolute error:",mean_absolute_error(y_test,y_pred7))
print("mean squared error:",mean_squared_error(y_test,y_pred7))
print("root mean squared error:",np.sqrt(mean_squared_error(y_test,y_pred7)))
print("R2 score:",r2_score(y_test,y_pred7))

# %% [markdown]
# **Comparative Analysis**

# %% [markdown]
# From the above results of models performances, Linear Regression is the most
# accurate model with lowest MSE, RMSE and MAE values and R-squared with value 1. This model has perfect accuracy for temperature rise predictions.
# Random forest slightly outperforms decision tree in terms of Mean Absolute Error but has a slightly higher Mean Squared Error and RMSE. 
# Random forest and decision tree both perform very well, with R-squared scores close to 1, indicating strong predictive power.

# %% [markdown]
# **Conclusion**

# %% [markdown]
# There are significant results and benefits from the findings of this project for environmental policy and strategic planning . The Linear regression model stands out as a reliable tool with its accurate prediction of temperature rise globally. This helps policymakers for forecasting and managing future temperature rise. Furthermore, the identification and implementation of effective strategies by stakeholders that target emissions can reduce global temperature increase. To conclude, this project successfully implemented data mining techniques for predictive analysis of temperature increase from CO2 emission from agri-food activities with the historic data from the globe. Further the project showed the relationship between CO2 emission, agri-food activities and temperature rise and identified Linear Regression as the most robust model in prediction of temperature rise in the agri-food sector. This implies that reliable predictive models are crucial for making successful policy and climate strategies.

# %%


# %%



