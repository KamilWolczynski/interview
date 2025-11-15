

!pip install awswrangler

# Cell 2 – Imports & session
import pandas as pd
import awswrangler as wr
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sklearn.model_selection import train_test_split

role = get_execution_role()
sess = sagemaker.Session()
region = boto3.Session().region_name

data_bucket = 'machine-learning-for-interview '
subfolder = 'chapter-02'
dataset = 'orders_with_predicted_value.csv'

# change
# df = pd.read_csv(f's3://{data_bucket}/{subfolder}/{dataset}')
# df.head()


# Cell 4 – Load data with AWS Data Wrangler (fast & IAM-aware)
df = wr.s3.read_csv(path=f"s3://{data_bucket}/{subfolder}/{dataset}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
df.head()

# 

# Cell 5 – Quick EDA (same as book)
print("\nTarget distribution:")
print(df["tech_approval_required"].value_counts())