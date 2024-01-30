import pandas as pd 
import numpy as np
from dotenv import load_dotenv
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import os
import io
from io import BytesIO
import google.generativeai as genai
load_dotenv()

# confirgure our API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# streamlit App
st.set_page_config(page_title = "CHAT TO DATA", layout="wide")
st.header("You can ask anything about below table")



fl = st.file_uploader(":file_folder: upload a file", type = (["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    file_content = fl.read()
    df = pd.read_csv(BytesIO(file_content), encoding="ISO-8859-1", parse_dates=True,infer_datetime_format=True, dayfirst=True)
    st.write(f"Uploaded file: {fl.name}")
    st.write(df.head())
else:
    df = pd.read_csv("Superstore.csv", encoding="ISO-8859-1")
    st.write("Using default file: Superstore.csv")
    st.write(df.head())

with st.expander("Expend to see the Statistics of Data"):
    st.write(df.describe(include='all'))

# Convert date columns to datetime
date_columns = [col for col in df.columns if 'date' in col.lower()]
df[date_columns] = df[date_columns].apply(pd.to_datetime, errors='coerce')



question = st.text_input("Input: ", key = "Input")
submit = st.button("Ask the question")

# Prompt for creating the Pandas code and dataframe

prompt_1=[
    f"""

    You are an expert in converting English questions to Pandas queries for any dataset! 
    Please adhere to the following guidelines for your responses: 
    The Pandas Dataframe has the name {df} and has the following columns - {df.columns} \n\n For example,

    Output format: Provide only the executable Pandas code, without at the beginning or end.
    
    \nExample 1 - How many entries of records are present?, 
    the Pandas code will be something like this 
    df.head(5)

    \nExample 2 - Tell me all the City wise sum of quantity?,
    the Pandas code will be something like this 
    (df.groupby(['City']).agg(
        Total_Quantity = ('Quantity','sum'))
        .sort_values('Total_Quantity',ascending = False)
        .reset_index())
    
    \nExample 3 - Provide me top 5 categories, 
    the Pandas code will be something like this 
    (df.groupby(['Category']).agg(
        Total_Quantity = ('Quantity', 'sum'))
        .nlargest(5,'Total_Quantity')
        .sort_values('Total_Quantity',ascending = False)
        .reset_index())
    
    \nExample 4 - Provide customers wise sum of sales,
    the Pandas code will be something like this 
    (df.groupby(['customers']).agg(
        Total_sales = ('sales','sum'))
        .sort_values('Total_sales',ascending = False)
        .reset_index())
    
    \nExample 5 - What is highest sales in one day?,
    the Pandas code will be something like this 
    
    (df.groupby(['Order Date']).agg(
        Total_sales = ('Sales', 'sum'))
        .nlargest(1)
        .reset_index())
    
    \nExample 6 - Provide me the order_status wise order_id count
    the pandas code will be something like this 
    (df.groupby(['order_status']).agg(
        Customer_count = ('order_id','count'))
        .sort_values('order_id',ascending= False)
        .reset_index())      
    
    \nExample 7 - Tell me the average sales for each city in the 'West' region.
    the Pandas code will be something like this 
    (df[df['Region'] == 'West'].groupby('City').agg(
        average_sales = ('sales','mean'))
        .sort_values('average_sales',ascending = False)
        .reset_index())  
    
    \nExample 8 - Add a new column 'Profit per Quantity' representing the profit earned per quantity sold.
    the pandas code will be something like this 
    df['Profit per Quantity'] = df['Profit'] / df['Quantity']
    
    \nExample 9 - Show me the total discount for each month in 2022.
    the pandas code will be something like this 
    (df[df['Order Date'].dt.year == 2022]
        .groupby(df['Order Date'].dt.to_period("M")).agg(
        Total_Discount = ('discount', 'sum'))
        .sort_values('Total_Discount',ascending = False)
        .reset_index())
        
    \nExample 10 - Show the cumulative sum of sales over time for the 'Technology' category.
    the pandas code will be something like this 
    (df[df['Category'] == 'Technology']
        .groupby('Order Date')['Sales']
        .cumsum())    
    
    \nExample 11 - Calculate the average sales for each day of the week.
    the pandas code will be something like this 
    (df.groupby(df['Order Date'].dt.day_name()).agg(
        average_sales = ('Sales','mean'))
        .sort_values('average_sales',ascending = False)
        .reset_index())
    
    \nExample 12 - Create a pivot table to display the average discount for each category and region combination.
    the pandas code will be something like this 
    ((df.pivot_table(
        index='Category',
        columns='Region',
        values='Discount',
        aggfunc='mean',
        fill_value=0)
        .reset_index())    
    
    \nExample 13 - Provide me the state wise customer count.
    the pandas code will be something like this 
    (df.groupby(['state']).agg(
        Customer_count = ('customer_id','count'))
        .sort_values('customer_id',ascending= False)
        .reset_index())    
    
    \nExample 14 - Provide me Product ID wise sum of Profit
    the pandas code will be something like this 
    (df.groupby(['Product ID']).agg(
        total_profit = ('Profit','sum'))
        .sort_values('Profit',ascending= False)
        .reset_index())
    
    \nExample 15 - How many unique customers are there?
    the Pandas code will be something like this 
    df['Customer ID'].nunique()

    \nExample 16 - Provide me "State" wise sum of "Quantity" and unique count of "Order ID", "Customer ID"
    the Pandas code will be something like this
    (df.groupby(['State']).agg(
        Total_Quantity = ('Quantity', 'sum'), 
        Unique_Order_ID = ('Order ID', 'nunique'),  
        Unique_Customer_ID = ('Customer ID', 'nunique'))
        .reset_index())

    \nExample 17 - Check for duplicate rows based on multiple columns: 
    the Pandas code will be something like this
    df.duplicated(subset=["customer_id", "order_id"], keep="first")

    \nExample 18 - Combine multiple columns into a single new column: 
    the Pandas code will be something like this
    df["Full Name"] = df["First Name"] + " " + df["Last Name"]

    \nExample 19 - Calculate conditional percentages:
    the Pandas code will be something like this
    df["Success Rate"] = (df["Successful Orders"] / df["Total Orders"]) * 100

    \nExample 20 - Provide me all the rows where "Electronics" in the "Category" column.
    the Pandas code will be something like this
    df["Category"].str.contains("Electronics", case=False)

    \nExample 21 - Filters for cities starting with the letter "L".
    the Pandas code will be something like this
    df["City"].str.startswith("L", case=False)

    \nExample 22 - Filters for names starting with "j" with case sensitivity.
    the Pandas code will be something like this
    df["Names"].str.startswith("j", case=True)
    
    \nExample 23 - Provide me city where count of "Order ID" is greater than 200
    the Pandas code will be something like this
    (df.groupby(['City']).agg(
        Unique_Order_ID = ('Order ID', 'nunique'))
        .query('Unique_Order_ID > 200')
        .reset_index())

    \nExample 24 - Using order_delivered_customer_date provide me monthly count of "order_id"
    the Pandas code will be something like this 
    (df['order_delivered_customer_date'].dt.to_period('M')
        .value_counts()
        .rename_axis('month')
        .reset_index(name='order_id_count'))

    \nExample 25 - Using "Order Date" provide me monthly sum of sales 
    the Pandas code will be something like this and sorting will be date or by month if we are grouping by any date
    (df.groupby(df['Order Date'].dt.to_period("M")).agg(
        Total_Sales = ('Sales', 'sum'))
        .sort_values('Order Date')
        .reset_index())

    \nExample 26 - using "Order Date" provide me yearly sum of "Quantity"
    the Pandas code will be something like this and sorting will be date or by month if we are grouping by any date
    (df.groupby(df['Order Date'].dt.year).agg(
        Total_Quantity = ('Quantity', 'sum'))
        .sort_values('Order Date')
        .reset_index())

    \nExample 27 - Using "Order Date" provide me monthly sum of "Quantity" and then create another column as cumulative sum on that column
    the Pandas code will be something like this and sorting will be date or by month if we are grouping by any date
    (df.groupby(df['Order Date'].dt.to_period("M")).agg(
        Total_Quantity = ('Quantity', 'sum'))
        .assign(Cumulative_Quantity = lambda x: x['Total_Quantity'].cumsum())
        .sort_values('Order Date')
        .reset_index())    
    
        
    \nExample 28 - Show me all orders placed between January 1st and March 31st, 2024.
    the Pandas code will be something like this and sorting will be date or by month if we are grouping by any date
    df[(df['Order Date'] >= '2024-01-01') & (df['Order Date'] <= '2024-03-31')]

    \nExample 29 - Find the average order processing time (difference between "Order Date" and "Ship Date")
    the Pandas code will be something like this
    df.assign(Processing_Time=df['Ship Date'] - df['Order Date'])['Processing Time'].mean()

    \nExample 30 - Provide me player wise sum of "wicket" where player == Yuvraj.
    the Pandas code will be something like this
    df[df['player'].isin(['Yuvraj'])].groupby('player')['wicket'].sum()
    
    
    and Pandas word in output also please make sure you are not including indexes in the final output"""]

# function to load google gemini model / that create the SQL query from the text
def get_gemini_response(quesiton, prompt_1):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([prompt_1[0],quesiton])
    return response.text

if submit:
    response = get_gemini_response(question, prompt_1)
    print(response)
    result_df = pd.DataFrame(eval(response))

    column_1 , column_2 = st.columns(2) 

    # download the data
    csv = result_df.to_csv(index = False).encode('utf-8')
    column_2.download_button("📥Download Data", data = csv, file_name= "Data.csv",
                            mime = "text/csv",
                            help = "Click here to download the data as a csv file")
 
    column_1.header(f"Total number of rows : {len(result_df)}") # Number of rows in the result
    
    # Result table and visualization

    # actual result
    
    formatted_df = result_df.head(10).style.format("{:.2f}", subset=pd.IndexSlice[:, result_df.select_dtypes(include=['float','int']).columns])
    
    st.table(formatted_df)
        
    # Pandas code
    st.subheader(f" Here is your Python-Pandas code \n {response}")
     
