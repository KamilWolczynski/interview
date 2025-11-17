## The old code needs adjustments
959c2f6a47b45c079111e8229350f58c0ee43856:Machine-learning-book/Readme.md

### Book was written in 2020.
I am going to adjust the code with Grok's help so I can run a test.

### Issue 1

```sh
from sklearn.model_selection import train_test_split  # didn't work because SageMaker Image doesn't include this package
```

quick fix:

```sh
!pip install scikit-learn
```
I was in a different SageMaker studio, which caused the packages issues. I finally found my way to an appropriate JupiterLab environment where all packages besides awswrangler were pre-installed.
**SageMaker notebook instance (Amazon Linux 2023, JupyterLab 4)**

### Issue 2: 403 Forbidden on pd.read_csv('s3://...')

```sh
df = pd.read_csv(f's3://{data_bucket}/{subfolder}/{dataset}')
```
Root Cause:

IAM Role lacks S3 permissions

quick fix:

Attached AmazonS3ReadFullAccess


Pro Tip: Book is Old → Modern Alternative

### Instead of s3fs + pd.read_csv
import awswrangler as wr --> is faster, more reliable, and handles permissions better.

```python
# Load data with AWS Data Wrangler (fast & IAM-aware)
!pip install awswrangler

df = wr.s3.read_csv(path=f's3://{data_bucket}/{subfolder}/{dataset}')
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
df.head()
```

### Issue 4: KeyError: "['tech_approval_required'] not found in axis"

```python
test_df = wr.s3.read_csv(path=f"s3://{data_bucket}/{processed_prefix}/test.csv")

# This returns a DataFrame with one column called '0', and the first row contains the entire CSV content as a string (starting with b'tech_approval_required,...)

```
```python
test_df.drop(columns=["tech_approval_required"])
```

### Output 

Columns in test_df:
['0']

First few rows:
                                                   0
0  b'tech_approval_required,role_tech,product_Cle...

### The bug:
```python
wr.s3.to_csv(pd.DataFrame([train_csv]), path=..., index=False, header=False)
```
This wraps your CSV string (train_csv) into a DataFrame with one row and one column.
Never wrap the CSV string in a DataFrame when uploading!
Instead, write the raw bytes directly to S3 using wr.s3.upload() or boto3.

### Issue 5: TypeError: upload() got an unexpected keyword argument 'content'
After editing the code 
```python
wr.s3.upload(content=train_csv, path=f"s3://{data_bucket}/{processed_prefix}/train.csv")
````
**awswrangler does not accept content= in upload()**

### Issue 6: TypeError: upload() got an unexpected keyword argument 'file_obj'
You're hitting this because awswrangler has changed its API over time — and you're likely using a newer version where wr.s3.upload() no longer supports file_obj= or content=

### the fix: using boto3
Use boto3.put_object() for raw bytes — it’s the most reliable.

```python
s3_client = boto3.client('s3')
processed_prefix = f"{subfolder}/processed"

s3_client.put_object(
    Bucket=data_bucket,
    Key=f"{processed_prefix}/train.csv",
    Body=train_csv
)
```

### Issue 7: ClientError: An error occurred (ValidationException) when calling the DeleteEndpoint operation: Could not find 
endpoint "order-approval-2025".

After uploading the files again, I needed to deploy the endpoint. I couldn't do that because I had already deleted it.


```python
endpoint_name = "order-approval-2025"

# Delete if already exists (optional)
try:
    sess.delete_endpoint(endpoint_name)
    print("Old endpoint removed")
except:
    pass
```
### Fix
Deleted the configuration of the endpoint and then deployed it again.

Before that, I wanted to know what endpoints and config exist.

```python
import boto3

sagemaker = boto3.client('sagemaker')
account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.session.Session().region_name

print(f"Account: {account_id} | Region: {region}\n")

# --- List all endpoints ---
print("ENDPOINTS:")
try:
    endpoints = sagemaker.list_endpoints()['Endpoints']
    if endpoints:
        for ep in endpoints:
            name = ep['EndpointName']
            status = ep['EndpointStatus']
            config = ep.get('EndpointConfigName', 'N/A')
            print(f"  - {name} | Status: {status} | Config: {config}")
    else:
        print("  No endpoints found.")
except Exception as e:
    print("  Error listing endpoints:", e)

print("\n" + "-"*60 + "\n")

# --- List all endpoint configs ---
print("ENDPOINT CONFIGS:")
try:
    configs = sagemaker.list_endpoint_configs()['EndpointConfigs']
    if configs:
        for cfg in configs:
            name = cfg['EndpointConfigName']
            created = cfg['CreationTime'].strftime("%Y-%m-%d %H:%M")
            print(f"  - {name} | Created: {created}")
    else:
        print("  No endpoint configs found.")
except Exception as e:
    print("  Error listing configs:", e)
```
output:

Account: 585008073988 | Region: us-east-1ENDPOINTS:
  No endpoints found.------------------------------------------------------------

ENDPOINT CONFIGS:order-approval-2025 | Created: 2025-11-15 14:21

### Issue 7: ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation: Received client error (415) 
from primary with message "Loading csv data failed with Exception, please ensure data is in csv format:
<class 'ValueError'>
**could not convert string to float: 'False'**

The XGBoost model tried to parse 'False' as a float, but it’s a string ('False') instead of 0.0 or 1.0.
XGBoost only accepts numeric CSV (no True/False, no strings).

### Root cause
My test_df has boolean columns

### Fix

```python
# CRITICAL: Convert ALL features to float
# This handles booleans, strings, etc.
X_test_numeric = X_test.astype(float).values
```
### Training data needed to be adjusted as well to float.

**<-- ADD .astype(float)**

```python
def to_sagemaker_csv(df, label_col, include_header=False):
    label = df[label_col].astype(int)
    feats = df.drop(columns=[label_col]).astype(float)  # <-- ADD .astype(float)
    out = pd.concat([label, feats], axis=1)
    return out.to_csv(None, header=include_header, index=False).encode('utf-8')
```

## Full Old vs. New Code Comparison

| **1. Load Data** | `pd.read_csv(f's3://...')` + `s3fs` | `awswrangler.s3.read_csv()` or fallback to `boto3` + `pd.read_csv()` | `s3fs` is slow, deprecated, and needs extra config. `awswrangler` is **AWS-native**, faster, IAM-aware. |


| **2. Feature Encoding** | `pd.get_dummies(df)` | Same | Logic unchanged — one-hot encoding is still valid. |


| **3. Target Column** | `corrs = encoded_data.corr()['tech_approval_required']` | `target = "tech_approval_required_1"` | After `get_dummies`, boolean → `0/1` → new column name. **You fixed this correctly!** |


| **4. Train/Val/Test Split** | `train_test_split(..., test_size=0.3)` | Same + `stratify=encoded[target]` | **Better**: `stratify` ensures class balance in splits → more reliable validation. |


| **5. Save to S3** | `with s3.open(...) as f: f.write(...)` | `boto3.put_object(Body=...)` | `awswrangler.to_csv(..., content=...)` **doesn’t exist**. `put_object()` is **simple, reliable, and works with bytes**. |


| **6. XGBoost Container** | `get_image_uri(..., 'xgboost', 'latest')` | `image_uris.retrieve("xgboost", version="1.7-1")` | `get_image_uri` **removed in SDK v2**. New API is required. |


| **7. Training Input** | `sagemaker.s3_input(...)` | `TrainingInput(...)` | `s3_input` **deprecated**. `TrainingInput` is the new standard. |


| **8. Instance Type** | `ml.m4.xlarge` | `ml.m5.xlarge` | `m4` **retired by AWS**. `m5` is current gen. |


| **9. Deploy** | `estimator.deploy(...)` + manual serializer | `estimator.deploy(..., serializer=CSVSerializer())` | Cleaner, explicit, and **avoids runtime errors**. |


| **10. Predict** | Manual `.decode('utf-8')` + split | `predictor.predict(X_test.values)` | Simpler, safer, batch-friendly. |

---

## Step-by-Step Explanation (Build Your Mental Model)

```python
# 1. Imports
import pandas as pd
import awswrangler as wr
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sklearn.model_selection import train_test_split
```

> **Why?**  
> - `awswrangler`: Best way to read/write S3 with **IAM role** (no keys).  
> - `TrainingInput`: SageMaker’s **new way** to point to training data.  
> - `CSVSerializer`: Ensures input to model is **clean CSV**.

---

```python
# 2. Load data
df = wr.s3.read_csv(path=f"s3://{data_bucket}/{subfolder}/{dataset}")
```

> **What happens?**  
> 1. `wr.s3.read_csv()` → calls `HeadObject` → **checks file exists**  
> 2. Downloads → parses as DataFrame  
> 3. Uses your **Studio execution role** → no access key needed


```python
# 3. One-hot encode
encoded = pd.get_dummies(df, drop_first=True)
```
> **Why?**  
> XGBoost can't handle strings like `"High"`, `"Low"`.  
> `get_dummies` → converts to `0/1` columns.


```python
# 4. Correlation filter
corrs = encoded.corr()[target].abs()
columns = corrs[corrs > 0.1].index.tolist()
encoded = encoded[columns]
```

> **Why?**  
> Remove noisy features.  
> Only keep features with **>10% correlation** to target → faster training, less overfitting.

---

```python
# 5. Split + Stratify
train_df, val_test_df = train_test_split(encoded, test_size=0.3, random_state=0, stratify=encoded[target])
val_df, test_df = train_test_split(val_test_df, test_size=0.333, random_state=0, stratify=val_test_df[target])
```

> **Why `stratify`?**  
> Ensures **same % of approvals** in train/val/test → **fair evaluation**.

---

```python
# 6. Convert to SageMaker format
def to_sagemaker_csv(df, label_col, include_header=False):
    label = df[label_col].astype(int)
    feats = df.drop(columns=[label_col])
    out = pd.concat([label, feats], axis=1)
    return out.to_csv(None, header=include_header, index=False).encode('utf-8')
```

> **SageMaker XGBoost expects**:
> - **First column = label** (`0` or `1`)
> - **No header** (except test)
> - **No index**

```python
# 7. Upload with boto3 (YOUR FIX!)
s3.put_object(Bucket=..., Key=..., Body=train_csv)
```

> **Why `awswrangler` failed?**
```python
wr.s3.to_csv(..., content=...)  # DOES NOT EXIST
```
> `to_csv()` only accepts **DataFrame + path**, not raw bytes.

> **Why `boto3.put_object()` works**:
> - `Body=` accepts **bytes** → perfect for `encode('utf-8')`
> - Simple, reliable, **no extra deps**

```python
# 8. TrainingInput
train_input = TrainingInput(s3_data=..., content_type="csv")
```

> Tells SageMaker:  
> “Here’s my training file. It’s CSV. Use it.”


```python
# 9. Train XGBoost
estimator.fit({"train": train_input, "validation": val_input})
```

> SageMaker:
> 1. Spins up `ml.m5.xlarge`
> 2. Downloads data
> 3. Trains XGBoost
> 4. Early stops if validation AUC stops improving
> 5. Saves model to S3

# 10. Deploy
> predictor = estimator.deploy(..., serializer=CSVSerializer())
> Creates real-time HTTPS endpoint
> Input: CSV row → Output: probability

# 11. Test
> predictions = predictor.predict(X_test.values)

> How testing works:X_test = features only (no label)
> predictor.predict() → sends to endpoint
> Returns probability (e.g., 0.72)
> You threshold at 0.5 → 1 = needs approval

Accuracy:
```python

(pred == true).mean() → 88%
```

→ Model is 88% correct on unseen data



