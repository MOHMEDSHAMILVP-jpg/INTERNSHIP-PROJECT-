#sql load to python
import pandas as pd
from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://root:""@localhost/sales_project")

query = "SELECT * FROM sales_dataset"
df = pd.read_sql(query, engine)
print(df.head())

#RFM calculation
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
snapshot_date = df["Order_Date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("Customer_ID").agg({
    "Order_Date": lambda x: (snapshot_date - x.max()).days,
    "Order_ID": "count",
    "Sales": "sum"
})

rfm.rename(columns={
    "Order_Date": "Recency",
    "Order_ID": "Frequency",
    "Sales": "Monetary"
}, inplace=True)

print(rfm.head())

#RFM scoring
rfm["R_score"] = pd.qcut(rfm["Recency"], 4, labels=[4,3,2,1])
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm["M_score"] = pd.qcut(rfm["Monetary"], 4, labels=[1,2,3,4])
rfm["RFM_Score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)

print(rfm.head())

#Customer segmentation
def segment_customer(row):

    if row["RFM_Score"] == "444":
        return "Best Customers"

    elif row["R_score"] == 4:
        return "Recent Customers"

    elif row["F_score"] == 4:
        return "Loyal Customers"

    elif row["M_score"] == 4:
        return "Big Spenders"

    else:
        return "Normal Customers"

rfm["Segment"] = rfm.apply(segment_customer, axis=1)

print(rfm.head())

#Market basket analysis
basket = df.groupby(["Order_ID","Product_Name"])["Sales"].sum().unstack().fillna(0)

print(basket.head())
basket = basket.map(lambda x: 1 if x > 0 else 0)
basket = basket.astype(bool)


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
frequent_items = apriori(basket, min_support=0.05, use_colnames=True)

print(frequent_items.head())


rules = association_rules(frequent_items, metric="lift", min_threshold=1)

print(rules.head())
rules.to_csv("market_basket_rules.csv", index=False)
print(rules.shape)
df.to_csv("sales_data.csv", index=False)

rfm.to_csv("rfm_customers.csv", index=False)

rules.to_csv("market_basket_rules.csv", index=False)

rfm.to_csv("rfm_customers.csv")
df.to_csv("sales_data_cleaned.csv")
rules.to_csv("market_basket_rules.csv")
