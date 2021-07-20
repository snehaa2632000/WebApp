import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px

def transform_data(raw_data):
    print(raw_data.columns)
    
    required_columns = ['Date', 'Permeate Flow (m3/hr)', 'Concentrate  Flow (m3/hr)','Feed Pressure (bar)', 'Concentrate Pressure (kg/cm2)',
                       'Permeate Pressure (kg/cm2)','Permeate Conductivity (µS/cm)','Feed Conductivity (µS/cm)','Feed TDS (mg/L)','Permeate TDS (mg/L)', 'Feed   Temp (F)', 'Feed   Temp (C)']
    
    data = raw_data[required_columns]

    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values(by=['Date'],inplace=True,ascending = True)
    data = data.fillna(0)

    # defining constants
    c3 = 8.01E-11
    c4 = 7.70E-20
    d3 = -50.64580519
    d4 = -90.47556224
    e3 = 112.483950289
    e4 = 188.88442227
    c11 = data["Concentrate  Flow (m3/hr)"][0]

    print(data.columns)
    print(data)
    
    # Calculated Temp (C)
    data.loc[data["Feed   Temp (F)"] > 0, "Calculated Temp (C)"] = (data["Feed   Temp (F)"]-32)*(5/9) 

    # Calculated Feed TDS (ppm)
    data.loc[data['Feed Conductivity (µS/cm)'] > 7630, "Calculated Feed TDS (ppm)"] = c3 * np.exp(((d3 - np.log(data['Feed Conductivity (µS/cm)']))**2)/e3)
    data.loc[(data['Feed Conductivity (µS/cm)'] >0) & (data['Feed Conductivity (µS/cm)'] <= 7630), "Calculated Feed TDS (ppm)"] = c4 * np.exp(((d4 - np.log(data['Feed Conductivity (µS/cm)']))**2)/e4)
    
    # Calculated Perm TDS
    data.loc[data['Permeate Conductivity (µS/cm)'] > 7630, "Calculated Perm TDS"] = c3 * np.exp(((d3 - np.log(data['Permeate Conductivity (µS/cm)']))**2)/e3)
    data.loc[(data['Permeate Conductivity (µS/cm)'] >0) & (data['Permeate Conductivity (µS/cm)'] <= 7630), "Calculated Perm TDS"] = c4 * np.exp(((d4 - np.log(data['Permeate Conductivity (µS/cm)']))**2)/e4)
    
    # Feed Flow (m3/hr)
    data.loc[data["Permeate Flow (m3/hr)"] > 0, "Feed Flow (m3/h)"] = data["Permeate Flow (m3/hr)"] + data["Concentrate  Flow (m3/hr)"]
    
    # Days of operation
    startup_date = data['Date'][0]
    data.loc[(data["Date"] - startup_date).dt.days > 0,"Days of Operation"] =(data["Date"] - startup_date).dt.days
    
    # Recovery (%)
    data.loc[data["Permeate Flow (m3/hr)"] > 0, "Recovery (%)"] = data["Permeate Flow (m3/hr)"] / data["Feed Flow (m3/h)"] 
    
    # Diferential Pressure
    data.loc[ data["Feed Pressure (bar)"] > 0 ,"Differential Pressure"] =  data["Feed Pressure (bar)"] - data["Concentrate Pressure (kg/cm2)"]

    # Temperature Correction Factor
    data.loc[data["Calculated Temp (C)"] > 25,"Temperature Correction Factor"]= np.exp(2640 * ((1 / 298.15) - 1 / (data["Calculated Temp (C)"] + 273.15)))
    data.loc[data["Calculated Temp (C)"] < 25,"Temperature Correction Factor"]= np.exp(3020 * ((1 / 298.15) - 1 / (data["Calculated Temp (C)"] + 273.15)))

    # Calculated Feed/Brine Avg Conc
    data.loc[data["Permeate Flow (m3/hr)"] > 0,"Calculated Feed/Brine Avg Conc"] = data["Feed TDS (mg/L)"]*(np.log(1/(1-(data["Recovery (%)"])))/(data["Recovery (%)"]))
    
    # Feed/Brine Osmotic Pressure (kPa)
    data.loc[ data["Feed Pressure (bar)"] > 0 ,"Feed/Brine Osmotic Pressure (kPa)"] = 0.0385*data["Calculated Feed/Brine Avg Conc"]*(data["Calculated Temp (C)"] +273.15)/(1000-(data["Calculated Feed/Brine Avg Conc"]/1000))/14.25
    
    #Permeate Osmotic Pressure
    data.loc[(data["Feed Pressure (bar)"] >0) & (data["Permeate Conductivity (µS/cm)"] > 0) & (data["Feed   Temp (F)"] > 0), "Permeate Osmotic Pressure"] =  0.0385 * data["Calculated Perm TDS"] * (data["Calculated Temp (C)"] + 273.15) / (1000 - (data["Calculated Perm TDS"] / 1000)) / 14.25
    data.loc[(data["Feed Pressure (bar)"] >0) & (data["Permeate Conductivity (µS/cm)"] > 0) & (data["Feed   Temp (F)"] < 0) & (data["Feed   Temp (C)"] > 0), "Permeate Osmotic Pressure"] =  0.0385 * data["Calculated Perm TDS"] * (data["Feed   Temp (C)"] + 273.15) / (1000 - (data["Calculated Perm TDS"] / 1000)) / 14.25
    data.loc[(data["Feed Pressure (bar)"] <0) & (data["Permeate TDS (mg/L)"] > 0) & (data["Feed   Temp (F)"] > 0) , "Permeate Osmotic Pressure"] =  0.0385 * data["Permeate TDS (mg/L)"] * (data["Calculated Temp (C)"] + 273.15) / (1000 - (data["Permeate TDS (mg/L)"] / 1000)) / 14.25
    data.loc[(data["Feed Pressure (bar)"] <0) & (data["Permeate TDS (mg/L)"] > 0) & (data["Feed   Temp (F)"] < 0) & (data["Feed   Temp (C)"] > 0), "Permeate Osmotic Pressure"] =  0.0385 * data["Permeate TDS (mg/L)"] * (data["Feed   Temp (C)"] + 273.15) / (1000 - (data["Permeate TDS (mg/L)"] / 1000)) / 14.25

    # Net Driving Pressure
    data.loc[data["Feed Pressure (bar)"] > 0,"Net Driving Pressure"]= data["Feed Pressure (bar)"]  - (data["Differential Pressure"] / 2) - data["Feed/Brine Osmotic Pressure (kPa)"] - data["Permeate Pressure (kg/cm2)"] + data["Permeate Osmotic Pressure"]
   
    x11 = data["Net Driving Pressure"][0]
    t11 = data["Temperature Correction Factor"][0]
    g11 = data["Calculated Feed TDS (ppm)"][0]
    b11 = data["Permeate Flow (m3/hr)"][0]
    u11 = data["Calculated Feed/Brine Avg Conc"][0]
    # Normalised Permeate Flow
    data.loc[data["Permeate Flow (m3/hr)"] > 0,"Normalized Permeate Flow"] = data["Permeate Flow (m3/hr)"] *x11*t11 / data["Net Driving Pressure"] / data["Temperature Correction Factor"]
    
    # Normalized Permeate  Salt Passage
    data.loc[(data["Permeate Flow (m3/hr)"] > 0) &(data["Feed Conductivity (µS/cm)"] > 0), "Normalized Permeate  Salt Passage"] = ((data["Calculated Perm TDS"] / data["Calculated Feed TDS (ppm)"]) * (data["Permeate Flow (m3/hr)"] *t11 *u11* data["Calculated Feed TDS (ppm)"]) / (b11 * data["Temperature Correction Factor"] * data["Calculated Feed/Brine Avg Conc"] * g11))*100
    data.loc[(data["Permeate Flow (m3/hr)"] > 0) &(data["Feed Conductivity (µS/cm)"] <0) & (data["Permeate TDS (mg/L)"] > 0), "Normalized Permeate  Salt Passage"] = ((data["Permeate TDS (mg/L)"] / data["Feed TDS (mg/L)"]) * (data["Permeate Flow (m3/hr)"] * t11 *u11 * data["Feed TDS (mg/L)"]) / (b11 * data["Temperature Correction Factor"] * data["Calculated Feed/Brine Avg Conc"] *t11))*100
    
    # Normalized Permeate  Salt Rejection
    data.loc[(data["Permeate Flow (m3/hr)"] > 0)&(data["Feed Conductivity (µS/cm)"] > 0),"Normalized Permeate  Salt Rejection"]= (100.0 - data["Normalized Permeate  Salt Passage"])
    data.loc[(data["Permeate Flow (m3/hr)"] > 0)&(data["Feed Conductivity (µS/cm)"] < 0)&(data["Feed TDS (mg/L)"]> 0),"Normalized Permeate  Salt Rejection"] = (100.0 - data["Normalized Permeate  Salt Passage"])
    
    # Normalized Pressure Differential
    data["Normalized Pressure Differential"]= data["Differential Pressure"]*((b11+c11)/2)/((data["Permeate Flow (m3/hr)"]+data["Concentrate  Flow (m3/hr)"]))/2

    #Rounding off the column values of calculated variables
    #for col in data.columns:
      #data[col] = data[col].round(2)
 
    
    #data.to_csv("final_ro.csv")
    return data.copy()


st.title("RO-AI")
st.markdown("Upload file")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  
  data = pd.read_csv(uploaded_file,encoding="unicode_escape")
  df = transform_data(data)
  df.to_csv(r'C:\Users\snehaa\Desktop\College\Buckman Miscellaneous\RO-AI\result_streamlit.csv', index=False)

  is_check = st.checkbox("Display Data")
  if is_check:
    st.write(df)
  
  dt1 = df[["Days of Operation","Normalized Permeate Flow"]]
  dt2 = df[["Days of Operation",'Differential Pressure']]
  dt3 = df[["Days of Operation","Normalized Permeate  Salt Passage"]]
  dt4 = df[["Days of Operation","Normalized Permeate  Salt Rejection"]]

  fig1 = px.line(dt1, x="Days of Operation", y="Normalized Permeate Flow", title="Normalized Permeate Flow",template="simple_white")
  st.plotly_chart(fig1)

  fig2 = px.line(dt2, x="Days of Operation", y="Differential Pressure", title="Differential Pressure",template="simple_white")
  st.plotly_chart(fig2)

  fig3 = px.line(dt3, x="Days of Operation", y="Normalized Permeate  Salt Passage", template="simple_white",title="Normalized Permeate  Salt Passage")
  fig3.update_yaxes(ticksuffix="%")
  st.plotly_chart(fig3)

  fig4 = px.line(dt4, x="Days of Operation", y="Normalized Permeate  Salt Rejection", template="simple_white",title="Normalized Permeate  Salt Rejection")
  fig4.update_yaxes(ticksuffix="%", showgrid=True)
  st.plotly_chart(fig4)