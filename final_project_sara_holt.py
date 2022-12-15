"""
Class: CS230--Section 4
Name: Sara Holt
Description:
This is my final project analyzing patterns of volcano eruptions around the world.
I pledge that I have completed the programming assignment independently.
I have not copied the code from a student or any source.
I have not given my code to any student.
"""
import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import folium
from streamlit_folium import folium_static
from matplotlib import colors

FILENAME = "Volcano_Eruptions.csv"


def readFunction(file):
    # Function to read in the given file as a data frame

    df = pd.read_csv(file, header=0,
                     names=['VolcanoNumber', 'VolcanoName', 'Country', 'PrimaryVolcanoType', 'ActivityEvidence',
                            'LastKnownEruption',
                            'Region', 'Subregion', 'Latitude', 'Longitude', 'Elevation(m)', 'DominantRockType',
                            'TectonicSetting'])
    return df


def mapFunction(dataframe):
    # Function to get map of volcano eruptions using the folium package
    mapData = dataframe[['VolcanoName', 'Latitude', 'Longitude']]

    # Asking for user input for zoom
    slider = st.slider('Select Zoom: ', 1, 25, 5)
    center = [38.789, 15.213]

    map1 = folium.Map(location=center, zoom_start=slider, control_scale=True)
    tooltips = 'Eruption'

    for index, location in mapData.iterrows():
        folium.Marker(location=[location["Latitude"], location["Longitude"]],
                      icon=folium.Icon(color='black', icon_color='red', icon="warning-sign"),
                      popup=location["VolcanoName"],
                      tooltip=tooltips).add_to(map1)
    folium_static(map1)


def groupByRegion(dataFrame):
    # Function grouping the data by region and counting the countries

    groupedDf = pd.DataFrame(dataFrame.groupby(['Region'])['Country'].count())
    groupedDf = groupedDf.reset_index()
    sortedDf = groupedDf.sort_values(by=['Region'], ascending=True)

    regionList = []
    for region in sortedDf.Region:
        if region not in regionList:
            regionList.append(region)
    return sortedDf, regionList


def regionChart(dataframe, region, color):
    # Function to get bar plot of countries and eruptions in selected region

    # Get data frame for specified region
    df = dataframe.query('Region == @region')

    # Sort the data frame by country
    sortedDf = df.sort_values(by=['Country'], ascending=True)

    # Get the number of eruptions per country by counting volcano numbers
    df2 = pd.DataFrame(df.groupby(['Country'])['VolcanoNumber'].count())

    # Get list of countries for x-axis labels
    countryList = []
    for country in sortedDf.Country:
        if country not in countryList:
            countryList.append(country)

    # Get the frequency for the y-axis
    eruptionsList = []
    for eruptions in df2.VolcanoNumber:
        eruptionsList.append(eruptions)

    # Create a horizontal bar chart - can change color based on user input
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.barh(countryList, eruptionsList, color=color, align='center', height=0.4)
    plt.ylabel("Countries in region", fontsize=14)
    plt.xlabel("No. of eruptions in country", fontsize=14)
    plt.title(f"Eruptions in different countries of the {region} region", fontsize=16, fontweight='bold')

    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    ax.grid(visible=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)
    return df2, fig


def regionChartAlaska(dataframe, color):
    # Func to get bar plot of subregions and eruptions in Alaska
    # (b/c country is US, only shows up as one bar)

    # Same code as above but using subregions of Alaska
    df = dataframe.query('Region == "Alaska"')
    sortedDf = df.sort_values(by=['Subregion'], ascending=True)
    df2 = pd.DataFrame(df.groupby(['Subregion'])['VolcanoNumber'].count())

    subregionList = []
    for subregion in sortedDf.Subregion:
        if subregion not in subregionList:
            subregionList.append(subregion)

    eruptionsList = []
    for eruptions in df2.VolcanoNumber:
        eruptionsList.append(eruptions)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.barh(subregionList, eruptionsList, color=color, align='center', height=0.4)
    plt.ylabel("Areas in subregion", fontsize=14)
    plt.xlabel("No. of eruptions in subregion", fontsize=14)
    plt.title(f"Eruptions in different subregions of the Alaska region", fontsize=16, fontweight='bold')

    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)
    return df2, fig


def groupByFeatures(dataFrame):
    # Function creating data frame to display certain features of the volcanoes - df to be used later
    df = dataFrame[[
        'VolcanoName', 'Region', 'Country', 'PrimaryVolcanoType', 'ActivityEvidence', 'LastKnownEruption',
        'Elevation(m)',
        'DominantRockType',
        'TectonicSetting']]

    # Create a list of possible volcano types
    volcano = []
    for types in df['PrimaryVolcanoType']:
        if types not in volcano:
            volcano.append(types)

    # Create a list of all possible dominant rock types
    rockType = []
    for rock in df['DominantRockType']:
        if rock not in rockType:
            rockType.append(rock)

    # Create a list of all possible tectonic settings
    tectonic = []
    for word in df['TectonicSetting']:
        if word not in tectonic:
            tectonic.append(word)

    # Used lists in order to get user input
    return df, volcano, rockType, tectonic


def typePieChart(dataFrame, volcanoType, color):
    # Function to get pie chart of all regions based on selected volcano type

    # First, get a dataframe with only the selected volcano type
    df = dataFrame[(dataFrame.PrimaryVolcanoType == volcanoType)]

    # Sort the values by region to get alphabetical order
    df = df.sort_values(by='Region', ascending=True)

    # Group the values by region and count the volcano numbers to get number of volcanoes per region
    df2 = pd.DataFrame(df.groupby(['Region'])['VolcanoNumber'].count())

    # Get a list of the regions that contain the selected volcano type
    regionsList = []
    for region in df.Region:
        if region not in regionsList:
            regionsList.append(region)

    # Get a list of the counts per each region of the selected volcano type
    countList = []
    for count in df2.VolcanoNumber:
        countList.append(count)

    # Create the pie chart with customized features
    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': color}

    def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(countList,
                                      autopct=lambda pct: func(pct, countList),
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    ax.legend(wedges, regionsList,
              title="Regions",
              loc="best",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Pie Chart of Regions by Primary Volcano Type")
    st.write(fig)


def allTypePieChart(dataframe, typeList, color):
    # Function to get pie chart displaying top 5 primary volcano types of data frame
    df = dataframe.sort_values(by='Region', ascending=True)

    # Group by primary volcano type and count the number of volcanoes in each type
    df2 = pd.DataFrame(df.groupby(['PrimaryVolcanoType'])['VolcanoNumber'].count())

    # Get only the top 5 primary volcano types
    df2 = df2.sort_values(by='VolcanoNumber', ascending=False)
    df3 = df2.head()

    countList = []
    for count in df3.VolcanoNumber:
        countList.append(count)

    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': color}

    def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(countList,
                                      autopct=lambda pct: func(pct, countList),
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    ax.legend(wedges, typeList,
              title="Volcano Types",
              loc="best",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Pie Chart of Eruptions by Top Five Volcano Type")
    st.write(fig)


def rockPieChart(dataframe, rockType, color):
    # Function to get pie chart of all regions based on rock type (same code as above except using rock type)
    df = dataframe[(dataframe.DominantRockType == rockType)]
    df = df.sort_values(by='Region', ascending=True)
    df2 = pd.DataFrame(df.groupby(['Region'])['VolcanoNumber'].count())

    regionsList = []
    for region in df.Region:
        if region not in regionsList:
            regionsList.append(region)

    countList = []
    for count in df2.VolcanoNumber:
        countList.append(count)

    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': color}

    def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(countList,
                                      autopct=lambda pct: func(pct, countList),
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    ax.legend(wedges, regionsList,
              title="Regions",
              loc="best",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Pie Chart of Regions by Dominant Rock Type")
    st.write(fig)


def allRockPieChart(dataframe, rockList, color):
    # Function to get pie chart of top 5 rock types of whole dataframe (same code as above except using rock type)
    df = dataframe.sort_values(by='Region', ascending=True)
    df2 = pd.DataFrame(df.groupby(['DominantRockType'])['VolcanoNumber'].count())
    df2 = df2.sort_values(by='VolcanoNumber', ascending=False)
    df3 = df2.head()

    countList = []
    for count in df3.VolcanoNumber:
        countList.append(count)

    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': color}

    def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(countList,
                                      autopct=lambda pct: func(pct, countList),
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    ax.legend(wedges, rockList,
              title="Dominant Rock Types",
              loc="best",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Pie Chart of Eruptions by Top Five Dominant Rock Types")
    st.write(fig)


def tectonicPieChart(dataframe, tectonicType, color):
    # Function to get pie chart of all regions based on tectonic setting (same as above except using tectonic setting)
    df = dataframe[(dataframe.TectonicSetting == tectonicType)]
    df = df.sort_values(by='Region', ascending=True)
    df2 = pd.DataFrame(df.groupby(['Region'])['VolcanoNumber'].count())

    regionsList = []
    for region in df.Region:
        if region not in regionsList:
            regionsList.append(region)

    countList = []
    for count in df2.VolcanoNumber:
        countList.append(count)

    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': color}

    def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(countList,
                                      autopct=lambda pct: func(pct, countList),
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    ax.legend(wedges, regionsList,
              title="Regions",
              loc="best",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Pie Chart of Regions by Tectonic Setting")
    st.write(fig)


def allTectonicPieChart(dataFrame, tectonicList, color):
    # Function to get pie chart of top 5 tectonic settings of whole data frame
    # (same code as above but with tectonic setting)
    df = dataFrame.sort_values(by='Region', ascending=True)
    df2 = pd.DataFrame(df.groupby(['TectonicSetting'])['VolcanoNumber'].count())
    df2 = df2.sort_values(by='VolcanoNumber', ascending=False)
    df3 = df2.head()

    countList = []
    for count in df3.VolcanoNumber:
        countList.append(count)

    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': color}

    def func(pct, allvalues):
        absolute = int(pct / 100. * np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(countList,
                                      autopct=lambda pct: func(pct, countList),
                                      startangle=90,
                                      wedgeprops=wp,
                                      textprops=dict(color="black"))

    ax.legend(wedges, tectonicList,
              title="Tectonic Settings",
              loc="best",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Pie Chart of Eruptions by Top Five Tectonic Settings")
    st.write(fig)


def customizeTypeTable(dataFrame, volcanoType="Maar(s)"):
    # Get table to show values of selected volcano type
    df = dataFrame[(dataFrame.PrimaryVolcanoType == volcanoType)]
    df = df.sort_values(by=['Region', 'Country'], ascending=True)
    return df


def customizeRockTable(dataFrame, rockType="Foidite"):
    # Get table to show values of selected rock type
    df = dataFrame[(dataFrame.DominantRockType == rockType)]
    df = df.sort_values(by=['Region', 'Country'], ascending=True)
    st.write(df)


def customizeTectonicTable(dataFrame, tectonicType):
    # Get table to show values of selected tectonic setting
    df = dataFrame[(dataFrame.TectonicSetting == tectonicType)]
    df = df.sort_values(by=['Region', 'Country'], ascending=True)
    st.write(df)


def pivotTable(dataFrame, selection):
    # Function creating pivot table displayed in Elevations page
    # Allows the user to choose tbl index based on their selection
    table = pd.pivot_table(dataFrame, values='Elevation(m)', index=[selection], columns=['Region'],
                           aggfunc=np.mean, fill_value=0)
    return table


def histogramElevation(dataFrame):
    # Get histogram of volcano elevations - ask user input for number of bins
    n_bins = st.slider('How many bins would you like?', 1, 25, 20)

    # Creating distribution for elevations on x-axis
    elevationsList = []
    for elevation in dataFrame['Elevation(m)']:
        elevationsList.append(elevation)

    x = np.array(elevationsList)

    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)

    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=10)

    axs.grid(visible=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)

    # Creating histogram
    N, bins, patches = axs.hist(x, bins=n_bins)
    fracs = ((N ** (1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    # Adding extra features
    plt.xlabel("Elevations", fontsize=14)
    plt.ylabel("Number of Volcano Eruptions", fontsize=14)
    plt.title('Histogram of Volcano Eruptions by Elevations', fontsize=16, fontweight='bold')

    # Show plot
    st.write(fig)


def datePlot(dataframe, color):
    # Get bar plot of eruptions per each time period

    # Get a list of all the dates with BCE - only get the digits
    bceDateList = []
    for date in dataframe['LastKnownEruption']:
        if 'BCE' in date:
            bceDateList.append(''.join(i for i in date if i.isdigit()))

    # Get a list of all the dates with CE - only get the digits
    ceDateList = []
    for date in dataframe['LastKnownEruption']:
        if 'CE' in date and 'B' not in date:
            ceDateList.append(''.join(i for i in date if i.isdigit()))

    # Get a list of all the unknowns
    unknownList = []
    for date in dataframe['LastKnownEruption']:
        if date == 'Unknown':
            unknownList.append(date)

    # Find the length of each of these lists to get the number of eruptions in BCE, etc.
    bceValues = len(bceDateList)
    ceValues = len(ceDateList)
    unknownValues = len(unknownList)

    # Create a vertical bar chart using the lengths
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(x=['BCE', 'CE', 'Unknown'], height=[bceValues, ceValues, unknownValues], color=color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(color)

    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)

    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    ax.set_xlabel('Time Period', labelpad=15, fontsize=14)
    ax.set_ylabel('Number of Eruptions', labelpad=15, fontsize=14)
    ax.set_title('Eruptions by Time Period', pad=15, weight='bold', fontsize=16)

    # Add text annotations to the top of the bars

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            round(bar.get_height(), 1),
            horizontalalignment='center',
            color=color,
            weight='bold'
        )

    fig.tight_layout()
    st.write(fig)


def time(dataframe, period, color, n_bins):
    # Get histogram of eruptions for selected time period

    # Take out all the dates that are unknown
    df = dataframe.query('LastKnownEruption != "Unknown"')

    # Get only the numbers for the dates (not including CE/BCE) and make them into integers
    # BCE dates have negative values, CE dates have positive values
    for i in range(len(df.LastKnownEruption)):
        if 'BCE' in df['LastKnownEruption'].iloc[i]:
            df['LastKnownEruption'].iloc[i] = int(
                ''.join(i for i in df['LastKnownEruption'].iloc[i] if i.isdigit())) * -1
        elif 'CE' in df['LastKnownEruption'].iloc[i] and 'B' not in df['LastKnownEruption'].iloc[i]:
            df['LastKnownEruption'].iloc[i] = int(''.join(i for i in df['LastKnownEruption'].iloc[i] if i.isdigit()))

    # Made them into integers in order to be able to sort them
    df2 = df.sort_values(by='LastKnownEruption', ascending=True)

    # Get the data frame of CE or BCE eruptions based on user input
    if period == 'BCE':
        df3 = df2.query('LastKnownEruption < 0')
    else:
        df3 = df2.query('LastKnownEruption > 0')

    # Group by last known eruption and count the number of volcanoes that erupted within that time period
    df4 = pd.DataFrame(df3.groupby(['LastKnownEruption'])['VolcanoNumber'].count())

    # Get a list of the dates to be the x-axis
    dateList = []
    for date in df3.LastKnownEruption:
        dateList.append(date)

    # Get the counts for y-axis
    countList = []
    for count in df4.VolcanoNumber:
        countList.append(count)

    # Creating histogram

    x = np.array(dateList)
    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)

    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=10)

    axs.grid(visible=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)

    # Creating histogram
    axs.hist(x, bins=n_bins, color=color)

    # Adding extra features
    plt.xlabel("Dates", fontsize=14)
    plt.ylabel("Number of Volcano Eruptions", fontsize=14)

    if period == 'BCE':
        plt.title('Histogram of Volcano Eruptions by Date (BCE)', fontsize=16, fontweight='bold')
    if period == 'CE':
        plt.title('Histogram of Volcano Eruptions by Date (CE)', fontsize=16, fontweight='bold')

    # Show plot
    st.write(fig)


def fullHistogramTime(dataframe, n_bins):
    # Get histogram of all time periods of eruptions
    df = dataframe.query('LastKnownEruption != "Unknown"')

    for i in range(len(df.LastKnownEruption)):
        if 'BCE' in df['LastKnownEruption'].iloc[i]:
            df['LastKnownEruption'].iloc[i] = int(
                ''.join(i for i in df['LastKnownEruption'].iloc[i] if i.isdigit())) * -1
        elif 'CE' in df['LastKnownEruption'].iloc[i] and 'B' not in df['LastKnownEruption'].iloc[i]:
            df['LastKnownEruption'].iloc[i] = int(''.join(i for i in df['LastKnownEruption'].iloc[i] if i.isdigit()))

    df2 = df.sort_values(by='LastKnownEruption', ascending=True)

    df3 = pd.DataFrame(df2.groupby(['LastKnownEruption'])['VolcanoNumber'].count())

    dateList = []
    for date in df2.LastKnownEruption:
        dateList.append(date)

    countList = []
    for count in df3.VolcanoNumber:
        countList.append(count)

    # Creating histogram
    x = np.array(dateList)
    fig, axs = plt.subplots(1, 1,
                            figsize=(10, 7),
                            tight_layout=True)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)

    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad=5)
    axs.yaxis.set_tick_params(pad=10)

    axs.grid(visible=True, color='grey',
             linestyle='-.', linewidth=0.5,
             alpha=0.6)

    # Creating histogram
    N, bins, patches = axs.hist(x, bins=n_bins)
    fracs = ((N ** (1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    # Adding extra features
    plt.xlabel("Dates", fontsize=14)
    plt.ylabel("Number of Volcano Eruptions", fontsize=14)
    plt.title('Histogram of Volcano Eruptions by All Dates', fontsize=16, fontweight='bold')

    # Show plot
    st.write(fig)


def main():
    st.sidebar.write("Explore the data!")
    choice = st.sidebar.radio("Choose an option to explore:",
                              ['Home Page', 'Eruptions Map', 'Volcanoes by Region', 'Volcano Features',
                               'Volcano Elevations', 'Eruption Dates'])
    color = st.sidebar.color_picker('Pick a color')
    volcano = readFunction(FILENAME)

    # Introduce to the data, print the doc statement, read in the file
    if choice == 'Home Page':
        st.markdown(f"<h1 style='text-align: center; color: {color};'>Volcano Eruptions Around the World</h1>",
                    unsafe_allow_html=True)
        st.write(
            "Welcome to a data exploration of volcano eruptions.  The data includes information on the volcano name, "
            "region, country, date of eruption, and volcano features.  Click on the radio buttons in the side bar to "
            "learn more about the patterns of these eruptions.")
        st.image('volcano.jpg')
        st.markdown(f"<sub style='text-align: center; '>{__doc__}</sub>",
                    unsafe_allow_html=True)

    # Show the folium map of eruptions
    elif choice == 'Eruptions Map':
        st.markdown(f"<h1 style='text-align: center; color: {color};'>Locations of Erupted Volcanoes</h1>",
                    unsafe_allow_html=True)
        mapFunction(volcano)

    # Ask for user input for region
    elif choice == 'Volcanoes by Region':
        volcanoRegion, regions = groupByRegion(volcano)
        st.markdown(f"<h1 style='text-align: center; color: {color};'>Erupted Volcanoes by Region and Country</h1>",
                    unsafe_allow_html=True)
        regionChoice = st.sidebar.selectbox("Choose a region", [elements for elements in regions])
        st.write(f"This is a bar chart that shows the number of eruptions by selected region and country.  "
                 f"Alaska is by subregion because the only country that shows is the US, therefore we can "
                 f"delve deeper into this region.")

        # Specify that the data will be different for Alaska
        if regionChoice != 'Alaska':
            regionData, chart = regionChart(volcano, regionChoice, color)
            st.write(chart)
        else:
            alaskaData, chart2 = regionChartAlaska(volcano, color)
            st.write(chart2)

    # Ask for user input of which volcano features they would like to explore
    elif choice == 'Volcano Features':
        st.markdown(f"<h1 style='text-align: center; color: {color};'>Explore Eruptions by Volcano Features</h1>",
                    unsafe_allow_html=True)
        featureDf, types, rocks, tectonics = groupByFeatures(volcano)
        feature = st.sidebar.radio('Choose feature to explore',
                                   ['Volcano Type', 'Dominant Rock Type', 'Tectonic Setting'])

        if feature == 'Volcano Type':
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Explore the Data by Primary Volcano Type</h2>",
                unsafe_allow_html=True)
            volcanoType = st.selectbox("Choose which primary volcano type", [x for x in types])
            st.write(customizeTypeTable(featureDf, volcanoType))
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Pie Chart by Region of Selected Volcano Type</h2>",
                unsafe_allow_html=True)
            typePieChart(volcano, volcanoType, color)
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Pie Chart of Eruptions by Top Five Volcano "
                f"Types</h2>",
                unsafe_allow_html=True)
            allTypePieChart(volcano, types, color)

        elif feature == 'Dominant Rock Type':
            st.markdown(f"<h2 style='text-align: center; color: {color};'>Explore the Data by Dominant Rock Type</h2>",
                        unsafe_allow_html=True)
            rockType = st.selectbox('Choose which dominant rock type', [x for x in rocks])
            customizeRockTable(featureDf, rockType)
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Pie Chart by Region of Selected Dominant Rock"
                f" Type</h2>",
                unsafe_allow_html=True)
            rockPieChart(volcano, rockType, color)
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Pie Chart of Eruptions by Top Five Dominant Rock "
                f"Types</h2>",
                unsafe_allow_html=True)
            allRockPieChart(volcano, rocks, color)

        else:
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Explore the Data by Tectonic Setting</h2>",
                unsafe_allow_html=True)
            tectonicType = st.selectbox('Choose which tectonic setting', [x for x in tectonics])
            customizeTectonicTable(featureDf, tectonicType)
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Pie Chart by Region of Selected Tectonic Setting"
                f"</h2>",
                unsafe_allow_html=True)
            tectonicPieChart(volcano, tectonicType, color)
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>Pie Chart of Eruptions by Top Five Tectonic Settings"
                f"</h2>",
                unsafe_allow_html=True)
            allTectonicPieChart(volcano, tectonics, color)

    # Will display a histogram of elevations and a pivot table
    elif choice == 'Volcano Elevations':
        featureDf, types, rocks, tectonics = groupByFeatures(volcano)
        st.markdown(f"<h1 style='text-align: center; color: {color};'>Explore by Elevations</h1>",
                    unsafe_allow_html=True)
        selectColumn = st.sidebar.selectbox('Choose how to index the data',
                                            ['PrimaryVolcanoType', 'ActivityEvidence', 'DominantRockType',
                                             'TectonicSetting'])
        st.markdown(
            f"<h2 style='text-align: center; color: {color};'>Pivot Table Displaying Average Elevation by Region and "
            f"{selectColumn}</h2>",
            unsafe_allow_html=True)
        st.write(pivotTable(featureDf, selectColumn))
        st.markdown(f"<h2 style='text-align: center; color: {color};'>Elevations Histogram</h2>",
                    unsafe_allow_html=True)
        histogramElevation(volcano)

    # Show two histograms of eruption dates - one based on user input, the other showing all time periods
    elif choice == 'Eruption Dates':
        st.markdown(f"<h1 style='text-align: center; color: {color};'>Explore Eruptions by Date</h1>",
                    unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: {color};'>Time Period Bar Chart</h2>",
                    unsafe_allow_html=True)
        # Shows bar chart
        datePlot(volcano, color)

        # Shows histogram of specific selected time period
        timePeriod = st.sidebar.selectbox('Select a time period', ['BCE', 'CE'])
        bins = st.slider('How many bins would you like?', 1, 25, 20)
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{timePeriod} Period Bar Chart</h2>",
                    unsafe_allow_html=True)
        time(volcano, timePeriod, color, bins)

        # Shows the histogram of ALL time periods
        st.markdown(f"<h2 style='text-align: center; color: {color};'>All Time Periods Histogram</h2>",
                    unsafe_allow_html=True)
        st.write('The dates from BCE are designated with a negative sign')
        fullHistogramTime(volcano, bins)


main()

# References for code above
# Histogram: https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
# Pie Chart: https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/
# Bar plot: https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
# Extracting digits from the eruption date: https://www.geeksforgeeks.org/python-extract-digits-from-given-string/
