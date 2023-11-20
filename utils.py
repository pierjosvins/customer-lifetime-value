import pandas as pd
import numpy as np

#from pyspark.ml.recommendation import ALSModel
#from pyspark.ml.feature import StringIndexerModel



def preprocess(original_data):
    data = original_data.dropna()
    data = data.drop_duplicates()
    data.CustomerID = data.CustomerID.astype(int)
    
    df = data[["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]]
    df.columns = ["CustomerID", "InvoiceNo", "Date", "Quantity", "UnitPrice"]
    
    df.loc[:, "Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df[df.Quantity > 0]
    
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    
    return df


def create_rfm_feature(df):
    max_date = df.Date.max()
    # Recency
    features_recency = df[["CustomerID", "Date"]]\
           .groupby(['CustomerID'])\
           .apply(lambda x: (x["Date"].max() - max_date) / pd.to_timedelta(1, "day"))\
           .to_frame().set_axis(["Recency"], axis=1)

    # Frequence
    features_frequence = df.drop_duplicates()[["CustomerID", "InvoiceNo", "Date"]]\
            .groupby(by=['CustomerID', 'InvoiceNo']).count()\
            .groupby(by=['CustomerID']).count().set_axis(["Frequence"], axis=1)
   
    # Monetary
    features_monetary = df\
            .groupby(by=['CustomerID']).agg({"Revenue": ["sum", "mean"]})\
            .set_axis(["MonetarySum", "MonetaryMean"], axis=1)

    features = pd.concat([features_recency, features_frequence,features_monetary], axis=1)
    return features

def predict_values(data, reg_model, classif_model):
    
    clv_pred = reg_model.predict(data)
    clv_pred = np.where(clv_pred < 0, 0, clv_pred)
    clv_prob = classif_model.predict_proba(data)
    data["PredictedCLV"] = clv_pred.flatten()
    data["ProbBuy"] = clv_prob[:, 1].flatten()
    data["ExpectedValue"] = data["PredictedCLV"] * data["ProbBuy"]
    return data.reset_index()


def make_recommendation(original_data, n=5):
    customer_ids = original_data.CustomerID.astype(int).astype(str).unique()
    #session = SparkSession.builder.remote("sc://localhost:8050").getOrCreate()
    session = SparkSession.builder.getOrCreate()

    model = ALSModel.load('data/recommendation_model')
    product_indexer = StringIndexerModel.load('data/product_indexer')
    customer_indexer = StringIndexerModel.load('data/customer_indexer')

    df_input = customer_indexer.transform(session.createDataFrame([(i,) for i in customer_ids], ["CustomerId"]))
    df_output = model.recommendForUserSubset(df_input, n)

    df_tmp_1 = df_output.select(df_output.CustomerIdIndex, f.explode(df_output.recommendations))
    df_tmp_2 = df_tmp_1.select('CustomerIdIndex', 'col.StockCodeIndex')
    customer_converter = IndexToString(inputCol="CustomerIdIndex", outputCol="CustomerId", labels=customer_indexer.labels)
    product_converter = IndexToString(inputCol="StockCodeIndex", outputCol="StockCode", labels=product_indexer.labels)
    df_output = customer_converter.transform(df_tmp_2)
    df_output = product_converter.transform(df_output)
    df_output = df_output.select(['CustomerId', 'StockCode'])
    df = df_output.toPandas()
    session.stop()

    product_infos = original_data[["StockCode", "Description"]]
    product_infos.Description = product_infos.Description.str.strip()
    product_infos = product_infos.drop_duplicates()
    product_infos.StockCode = product_infos.StockCode.astype(str)

    df = df.merge(product_infos, on="StockCode", how="inner").drop_duplicates()
    return df
