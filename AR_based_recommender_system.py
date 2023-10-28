'''Turkey's largest online service platform, Armut brings together service providers and those who want to receive
services. It provides easy access to services such as cleaning, renovation, transportation, etc. with a few taps on
your computer or smartphone. It is intended to create a product recommendation system withAssociationRuleLearning by
using the dataset containing the service users and the services and categories that the users have received.'''
import pandas as pd

'''Data: The data set consists of the services and categories of services received by customers. It contains the date 
and time of each service received'''

'''Task 1: Data Preparation
Step 1: Read armut_data.csvd file'''
# !pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 5/ArmutARL-221114-234936/armut_data.csv")
df.head()

'''Step 2: ServiceID represents a different service for eachCategoryID.  Create a new variable to represent these 
services by concatenating ServiceID andCategoryID with "_"'''

df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)
df["Service"]

'''Step 3: The data set consists of the date and time when the services were received, there is no basket definition 
(invoice, etc.). In order to apply Association Rule Learning, a basket definition (invoice, etc.) needs to be 
created. Here, the basket definition is the services purchased monthly by each customer. For example; 9_4, 
46_4 services received by the customer with id 7256 in the 8th month of 2017 refer to one basket; 9_4, 38_4 services 
received in the 10th month of 2017 refer to another basket. Baskets need to be identified with a unique ID. For this, 
first create a new date variable with year and month. Concatenate UserID and the newly createddate variable with "_" 
to create a new variable namedID.'''

df["CreateDate"] #dtype: object, lets convert it to date.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["New_Date"] =  df["CreateDate"].dt.strftime("%Y-%m")
df.info()

df["Basket_Id"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)
df.head()

'''Task 2: Produce and Suggest Association Rules
Step 1: Create a service pivot table as follows.'''

'''pivot_table = df.pivot_table(index='Basket_Id', columns='Service', values='ServiceId', aggfunc='count', fill_value=0)
pivot_table = pivot_table.map(lambda x: 1 if x > 0 else 0)
pivot_table'''

apriori_df = df.groupby(["Basket_Id", "Service"])["CategoryId"].count().unstack().fillna(0)
apriori_df=apriori_df.map(lambda x: 1 if x > 0 else 0)
apriori_df
'''Step 2: Create the rules of association.'''

from mlxtend.frequent_patterns import apriori, association_rules
association_test = apriori(apriori_df, min_support=0.01, use_colnames=True)
association_test

rules = association_rules(association_test, metric="lift", min_threshold=0.01)
rules

def gathered_data_test(dataframe):
    dataframe["Service"] = (dataframe["ServiceId"].astype(str) + "_" + dataframe["CategoryId"].astype(str))
    dataframe["CreateDate"] = pd.to_datetime(dataframe["CreateDate"])
    dataframe["New_Date"] = dataframe["CreateDate"].dt.strftime('%Y-%m')
    dataframe["SepetId"] = dataframe["UserId"].astype(str) + "_" + dataframe["New_Date"].astype(str)
    return dataframe
gathered_data_test(df)


def arl_recommender_test(dataframe):
    dataframe = gathered_data_test(dataframe)
    apriori_df = dataframe.groupby(['Basket_Id', 'Service'])['CategoryId'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    frequent_itemsets = apriori(apriori_df, min_support=0.01, use_colnames=True)
    rules = association_rules(association_test, metric="support", min_threshold=0.01)
    return rules

rules = gathered_data_test(df)


'''Step3: Use the arl_recommender function to recommend a service to a user who received the last 2_0 service'''


def arl_recommender(rules_df, service, rec_count=1):
    # Sort the rules based on lift in descending order
    sorted_rules = rules_df.sort_values("lift", ascending=False)

    # Filter the rules where the antecedent contains the given service
    filtered_rules = sorted_rules[sorted_rules["antecedents"].apply(lambda x: service in x)]

    # Extract the recommended services from the filtered rules
    recommendation_list = [', '.join(rule) for rule in filtered_rules["consequents"].apply(list).tolist()[:rec_count]]

    return recommendation_list[0:rec_count]

arl_recommender(rules,"2_0",1)

def arl_recommender1(rules_df, service_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list =[]
    for i, service in enumerate(sorted_rules["antecedents"]):
      for j in list(service):
          if j == service_id:
             recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules,"2_0",1)
