#%% [markdown]
# # Analyzing NYC High School Data
# ---
# This is a guided project part of the Dataquest.io courses. The goal of this project is to examine NYC high school data to find out what causes high SAT scores.
# 
# There is a good amount of public data available regarding education statistics and SAT scores. The only issue is that these are available in separate sources. Therefore the first part of this project we will need to read in the data and clean it up enough so that all of the datasets can be combined.
# 
# Additionally, we will also experiment with using the Altair library for visualizations.
#%% [markdown]
# # Read in the data

#%%
import pandas as pd
import numpy as np
import re
import matplotlib
import altair as alt
import geopandas as gpd
import gpdvega

data_files = [
    "ap_2010.csv",
    "class_size.csv",
    "demographics.csv",
    "graduation.csv",
    "hs_directory.csv",
    "sat_results.csv"
]

data = {}

for f in data_files:
    d = pd.read_csv("data/schools/{0}".format(f))
    data[f.replace(".csv", "")] = d

#%% [markdown]
# # Read in the surveys

#%%
all_survey = pd.read_csv("data/schools/survey_all.txt", delimiter="\t", encoding='windows-1252')
d75_survey = pd.read_csv("data/schools/survey_d75.txt", delimiter="\t", encoding='windows-1252')
survey = pd.concat([all_survey, d75_survey], axis=0, sort=False)

survey["DBN"] = survey["dbn"]
sruvey_fields = survey.columns

survey = survey.loc[:,survey_fields]
data["survey"] = survey

#%% [markdown]
# # Add DBN columns
#%% [markdown]
# The DBN column is the key field that will be used to join the various datasets together.

#%%
data["hs_directory"]["DBN"] = data["hs_directory"]["dbn"]

def pad_csd(num):
    string_representation = str(num)
    if len(string_representation) > 1:
        return string_representation
    else:
        return "0" + string_representation
    
data["class_size"]["padded_csd"] = data["class_size"]["CSD"].apply(pad_csd)
data["class_size"]["DBN"] = data["class_size"]["padded_csd"] + data["class_size"]["SCHOOL CODE"]

#%% [markdown]
# # Convert columns to numeric

#%%
cols = ['SAT Math Avg. Score', 'SAT Critical Reading Avg. Score', 'SAT Writing Avg. Score']
for c in cols:
    data["sat_results"][c] = pd.to_numeric(data["sat_results"][c], errors="coerce")

data['sat_results']['sat_score'] = data['sat_results'][cols[0]] + data['sat_results'][cols[1]] + data['sat_results'][cols[2]]

def find_lat(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lat = coords[0].split(",")[0].replace("(", "")
    return lat

def find_lon(loc):
    coords = re.findall("\(.+, .+\)", loc)
    lon = coords[0].split(",")[1].replace(")", "").strip()
    return lon

data["hs_directory"]["lat"] = data["hs_directory"]["Location 1"].apply(find_lat)
data["hs_directory"]["lon"] = data["hs_directory"]["Location 1"].apply(find_lon)

data["hs_directory"]["lat"] = pd.to_numeric(data["hs_directory"]["lat"], errors="coerce")
data["hs_directory"]["lon"] = pd.to_numeric(data["hs_directory"]["lon"], errors="coerce")

#%% [markdown]
# # Condense datasets

#%%
class_size = data["class_size"]
class_size = class_size[class_size["GRADE "] == "09-12"]
class_size = class_size[class_size["PROGRAM TYPE"] == "GEN ED"]

class_size = class_size.groupby("DBN").agg(np.mean)
class_size.reset_index(inplace=True)
data["class_size"] = class_size

data["demographics"] = data["demographics"][data["demographics"]["schoolyear"] == 20112012]

data["graduation"] = data["graduation"][data["graduation"]["Cohort"] == "2006"]
data["graduation"] = data["graduation"][data["graduation"]["Demographic"] == "Total Cohort"]

#%% [markdown]
# # Convert AP scores to numeric

#%%
cols = ['AP Test Takers ', 'Total Exams Taken', 'Number of Exams with scores 3 4 or 5']

for col in cols:
    data["ap_2010"][col] = pd.to_numeric(data["ap_2010"][col], errors="coerce")

#%% [markdown]
# # Combine the datasets

#%%
combined = data["sat_results"]

combined = combined.merge(data["ap_2010"], on="DBN", how="left")
combined = combined.merge(data["graduation"], on="DBN", how="left")

to_merge = ["class_size", "demographics", "survey", "hs_directory"]

for m in to_merge:
    combined = combined.merge(data[m], on="DBN", how="inner")

combined = combined.fillna(combined.mean())
combined = combined.fillna(0)

#%% [markdown]
# # Add a school district column for mapping

#%%
def get_first_two_chars(dbn):
    return dbn[0:2]

combined["school_dist"] = combined["DBN"].apply(get_first_two_chars)

#%% [markdown]
# # Find correlations

#%%
correlations = combined.corr()
correlations = correlations["sat_score"].abs()
print(correlations.sort_values(ascending=False).head(20))

#%% [markdown]
# Taking the absolute value of coefficient, we can get a quick snapshot of how each value correlates with SAT scores. Keep in mind that some of these may be negatively correlated since we took the absolute value. Some quick notes:
# - Scores on the individual portions correlate highly, but is not useful to us since the individual portions make up the total SAT score
# - A high number of students will naturally lead to higher scores due to more chances for a high scores to occur
#%% [markdown]
# # Plotting survey correlations

#%%
# Remove DBN since it's a unique identifier, not a useful numerical value for correlation.
survey_fields.remove("DBN")


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
combined.corr()["sat_score"][survey_fields] #.plot.bar()


#%%
survey_fields_corr = pd.DataFrame(combined.corr()['sat_score'][survey_fields]).reset_index()
survey_fields_corr = survey_fields_corr.rename(columns={'index': 'field', 'sat_score':'sat_score_corr'})
alt.Chart(survey_fields_corr).mark_bar().encode(
    x='field',
    y='sat_score_corr'
)

#%% [markdown]
# This a bar graph that compares SAT scores with responders on various school surveys. For example, the comp_p_11 column means the communication score from parents who answered the survey. In this case, where the communication score was reated poorly, this indicates a very loose negative correlation with SAT score.
# 
# The highest two values are the N_s and the N_p columns, which are the number of student and parent respondents respectively. It should be expected that these would be high given that these fields are related to student enrollment. The next highest are the aca_s_11, saf_s_11, saf_t_11, and saf_tot_11 columns, the _saf_ columns means Safety and Respect. _Aca_ means academic expectations.
# 
# There appears to be a definitive positive correlation between the scores and this Safety/Respect category. For academic expectations, the response is higher for students rather than teachers. Overall correlations howevwer, are fairly weak.

#%%
alt.Chart(combined).mark_point().encode(
    x=alt.X('saf_s_11:Q', 
           scale=alt.Scale(zero=False)),
    y=alt.Y('sat_score:Q',
           scale=alt.Scale(zero=False))
)

#%% [markdown]
# There appears to be a weak positive correlation between safety score and SAT score. We can generally see that a low safety score will result in a low SAT score. A high safety score however, has a weaker correlation and less likely to result in a high SAT score.
# 
# Note that Altair by default will always include the zero axis, even if there are no values near it. This can be disabled with the code included.
#%% [markdown]
# ## Visualising Safety Scores Geographically
#%% [markdown]
# We'll use GeoPandas to hold the geographical data needed to map the scores against school district boudnaries in New York City. GeoPandas is a Python library that extends support for geographical dimensions that allow for spatial mapping.
# 
# Two of the data types that Geopandas introduces are a _Point_ and _Polygon_. A point represents a coordinate pair in the format of longitude and latitude. The coordinates in the existing data frame, although it has latitidue and longitude pairs that we need is not in the correct format required by Geopanads. To convert, we'll use the Shaprely library to convert this into a _Point_.
# 
# When working with geodata, it's important to keep the _Coordinate Reference System_ or CRS in mind. When mapping multiple datasets together, they must all be on the same CRS. If they are not, then your mapping library may not display anything, or some of the data points may not line up correctly.
# 
# Finding and converting the CRS is relatively simple. GeoSeries and GeoDataFrames have a CRS property. Converting is also similarily straightforward. Use the `to_crs()` method, and pass in a dictionary in the format of `{'init': 'epsg:2263'}` to convert.
# 
# CRS codes can be researched using this site: https://www.epsg-registry.org
#%% [markdown]
# ### A note about mapping with Altair/Vega
# Mapping with GeoPandas directly onto matplotlib works great since everything is integrated nicely.
# 
# Mapping a GeoDataFrame with Altair was a different story unfortunately. Being based on Vega, data must be passed in as GeoJSON. While geopandas has a convenient `to_json()` method, there were several observed problems preventing the data from cleanly mapping.
# 
# After some searching, there is an additional library, `gpdvega` that bridges geopandas and Altair. Once the library is imported, you can use a GeoDataFrame directly in the Altair syntax and it works like magic.
# 
# Links for reference:
# - [Issue reported on Altair's github](https://github.com/altair-viz/altair/issues/588)
# - [`gpdvega` github](https://github.com/altair-viz/altair/issues/588)

#%%
safs_by_dist = combined.groupby('school_dist')[['lat','lon','saf_s_11']].mean()


#%%
nysd = gpd.read_file('shapes/nysd/nysd.shp')
nysd = nysd.to_crs({'init': 'epsg:4326'})
safs_by_dist.index = pd.to_numeric(safs_by_dist.index)
geo_safs_by_dist = nysd.join(safs_by_dist, how='inner')
geo_safs_by_dist = geo_safs_by_dist.drop(['lat','lon'], axis=1)
geo_safs_by_dist = geo_safs_by_dist.to_crs({'init': 'epsg:4326'})


#%%
# gpdvega library enables using a GeoDataFrame directly with Altair
alt.Chart(geo_safs_by_dist).mark_geoshape().encode(
    color=alt.Color('saf_s_11:Q', scale=alt.Scale(scheme='blues')),
    tooltip='SchoolDist:N'
).properties(
    width=500,
    height=500,
    title='Safety Score by NYC School District'
)

#%% [markdown]
# Darker shades show a higher score, while lighter scores show a lower score. We can see some parts of the Bronx and Queens that have higher scores.
#%% [markdown]
# ## Visualising Relation Between Ethnicity and Score

#%%
ethnicities = ['white_per','asian_per','black_per','hispanic_per']
eth_corr = pd.DataFrame(combined.corr()['sat_score'][ethnicities]).reset_index()
eth_corr = eth_corr.rename(columns={'index': 'ethnicity_per', 'sat_score': 'sat_score_corr'})
alt.Chart(eth_corr).mark_bar().encode(
    alt.X('ethnicity_per:N'),
    alt.Y('sat_score_corr:Q', scale=alt.Scale(domain=(-.9,.9)))
).properties(width=200)

#%% [markdown]
# A quick bar plot shows us that whites and Asians tend to have higher SAT scores, while blacks and Hispanics have a negative correlation with SAT score, although for the latter two, the correlation coefficient is not high. This could be due to a lack of funding for schools in certain neighborhoods that are more likely to have a higher percentage of black or hispanic students.

#%%
alt.Chart(combined).mark_point().encode(
    x='hispanic_per',
    y='sat_score'
)

#%% [markdown]
# Plotting Hispanics againist SAT score shows us a negative, non-linear relationship between Hispanics in a school versus SAT Score.

#%%
combined.columns[151:200]


#%%
cols_of_interest = ['school_name', 'boro', 'school_type', 'total_students','hispanic_per', 'sat_score']
combined[combined['hispanic_per'] > 95][cols_of_interest]

#%% [markdown]
# There are only eight schools with a 95% Hispanic students. Of note with these schools is that they are all relatively small schools. About half of these schools are specialised schools. Potentially for ESL students as suggested by some of the names.

#%%
cols_of_interest = ['school_name', 'boro', 'school_type', 'total_students', 'hispanic_per', 'sat_score']
combined[(combined['hispanic_per'] < 10) & (combined['sat_score'] > 1800)][cols_of_interest]

#%% [markdown]
# On the opposite end, looking at the top schools by score listed, these are all specialised schools as defined by the `school_type` field. Coincidentally, there are five schools that match the criteria and each borough is represented by one school in the list.
#%% [markdown]
# ## Visualising Correlations Between Gender and SAT Score

#%%
gender_corr = combined.corr()['sat_score'][['male_per', 'female_per']].reset_index()
gender_corr = gender_corr.rename(columns={'index': 'gender_per', 'sat_score':'sat_score_corr'})
alt.Chart(gender_corr).mark_bar().encode(
    alt.X('gender_per:N'),
    alt.Y('sat_score_corr:Q', scale=alt.Scale(domain=(-1,1)))
).properties(width=150)

#%% [markdown]
# Plotting the correlation between genders and scores, we see that females have a higher correlation. However, as indicated by the correlation coefficients, both are weakly correlated.

#%%
alt.Chart(combined).mark_point().encode(x='female_per', y='sat_score')

#%% [markdown]
# A scatter plot between these two columns shows us that there isn't much of a correlation between score and the percent of girls in schools. We can see that most schools generally have a even gender ratio. One interesting observation we can see is that the highest scores look like they come from schools with balanced gender ratios.
# 
# Note that in this dataset, there are only two schools where the sum of the male_per and female_per columns do not equal 100.

#%%
g = combined['male_per'] + combined['female_per']
g.value_counts()


#%%
combined[(combined['female_per'] > 60) & (combined['sat_score'] > 1700)][cols_of_interest]

#%% [markdown]
# A closer examination of the majority female schools with high SAT scores reveals that the schools listed here appear to be regarded highly in academics. These schools are highly selective and many of their students continue to high school.
#%% [markdown]
# ## AP Exams Impact on SAT Scores
# The next item examined is how AP Exams may affect SAT scores. We are expecting that schools with more AP test takers will have higher average scores. Since the total enrollment of scores tends has a strong correlation with the average SAT score for a school, we will examine the percent of AP test takers in a school instead, similar to how we examined the ratios of students in prior sections.

#%%
combined['ap_per'] = combined['AP Test Takers '] / combined['total_enrollment']
alt.Chart(combined).mark_point().encode(x='ap_per', y='sat_score')

#%% [markdown]
# Examining the scatter plot of the percent of students that take AP exams versus SAT Score, we do see some evidence that it positively affects SAT scores. However, we can also see that the schools with the highest percentages do not show a higher score. 
#%% [markdown]
# ## Conclusion
# This concludes this explorartory analysis of this data. We examined the data to see if there is any correlations with SAT scores. Looking for correlations and patterns can be helpful in identifying useful features for training models in machine learning.
# 
# Below are potential next steps to explore
# - Examine correlation between class size and SAT scores
# - Figure out which neighborhoods have the best schools
# - Investigate differences between teacher, parent, and student responses to the surveys
# - Classifying schools based on SAT scores and other attributes (this starts to dip into ML territory)

