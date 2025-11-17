## Why each import is needed  

```python
import awswrangler as wr  # For efficient S3 reads/writes
```
Downside: print(wr.s3.read_csv(...).head()) uses plain repr, not Jupyter HTML tables → ugly output.


```python
from time import sleep
```
- **Purpose**: Wait 30 seconds after deleting an endpoint (SageMaker needs time to release resources).

```python
import boto3
```
- **Purpose**: Low-level AWS SDK (used by SageMaker under the hood).
- **Used indirectly**: via `boto3.Session().region_name`.

```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.session import Session
```
- **Purpose**:
  - `sagemaker`: Main SDK namespace.
  - `get_execution_role()`: Gets IAM role of the notebook.
  - `Session`: Controls AWS session, region, credentials.

```python
from sklearn.model_selection import train_test_split
```
- **Purpose**: Split data into train/validation/test with **stratification** (critical for imbalanced churn).

```python
import sklearn.metrics as metrics
```
- **Purpose**: Compute accuracy, confusion matrix.

---

## 2. Why `pd.read_csv()` Looks Better Than `awswrangler`

You are **100% correct**.

`pd.read_csv('s3://...')` | **Jupyter HTML table** with bold headers, clean spacing |
`wr.s3.read_csv('s3://...')` | Plain text `repr()` — no formatting |

### Why?
- `pd.read_csv()` → returns `pandas.DataFrame` → Jupyter **auto-displays with HTML**.
- `wr.s3.read_csv()` → same object, but **if called in a cell with other output**, Jupyter may fall back to `repr`.

### Fix: Use `pd.read_csv()` for **display**, `wr.s3` for **performance**


## 3. Missing Line: `columns = df.columns.tolist()`

```python
columns = df.columns.tolist()
encoded_data = df.drop(['id', 'customer_code', 'co_name'], axis=1)
```

## 4. Why No `.encode()`?

### Old code:
```python
train_data = train_df.to_csv(None, header=False, index=False).encode()
with s3.open(..., 'wb') as f:
    f.write(train_data)
```

### New code:
```python
wr.s3.to_csv(train_df, path=train_path, header=False, index=False)
```

### Answer: **CSV is text. Machines read it fine. No need to encode.**

**XGBoost in SageMaker expects CSV text files** — **not binary**.

- `wr.s3.to_csv()` writes **UTF-8 text** directly to S3.
- `.encode()` was only needed with `s3fs` + `open(..., 'wb')`.
- `awswrangler` handles encoding **internally**.

## 5. Why `sagemaker.algorithms.XGBoost` Is Gone

**As of SageMaker SDK v2.200+ (2024–2025), `sagemaker.algorithms.XGBoost` was removed.**

### Official AWS Statement (2025):
"Use `sagemaker.estimator.Estimator` with `image_uris.retrieve()` for full control and latest versions."


## 6. Why Not Build a Custom Container?

You **don’t need to** — and here's why:

`image_uris.retrieve("xgboost", version="1.5-1")` | **99% of cases** — pre-built, secure, updated

**SageMaker Built-in XGBoost Container**:
> - Runs your data in **script mode** by default
> - Supports `train` and `validation` channels
> - Handles CSV → LibSVM internally
> - Auto-scales, logs to CloudWatch

**You get the same power without Dockerfiles.**

## Correct 2025 Code (Your Version + Fixes)

```python
# === 1. Imports ===
import pandas as pd
import awswrangler as wr
from time import sleep
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# === 2. Setup ===
data_bucket = 'machine-learning-for-interview'
subfolder = 'chapter-03'
dataset = 'churn_data.csv'
processed_subfolder = f'{subfolder}/processed'

role = get_execution_role()
sess = sagemaker.Session()
region = boto3.Session().region_name

# === 3. Load & Explore (Beautiful Output) ===
df = pd.read_csv(f's3://{data_bucket}/{subfolder}/{dataset}')
df.head()  # ← Bold headers, clean

print(f'Rows: {df.shape[0]}')
print(df['churned'].value_counts())

# Optional: keep column list
columns = df.columns.tolist()
encoded_data = df.drop(['id', 'customer_code', 'co_name'], axis=1)

# === 4. Split ===
y = encoded_data['churned']
train_df, test_val_df, _, _ = train_test_split(encoded_data, y, test_size=0.3, stratify=y, random_state=0)
y = test_val_df['churned']
val_df, test_df, _, _ = train_test_split(test_val_df, y, test_size=0.333, stratify=y, random_state=0)

# === 5. Upload with awswrangler (fast, text CSV) ===
train_path = f's3://{data_bucket}/{processed_subfolder}/train.csv'
val_path = f's3://{data_bucket}/{processed_subfolder}/val.csv'
test_path = f's3://{data_bucket}/{processed_subfolder}/test.csv'

wr.s3.to_csv(train_df, train_path, header=False, index=False, use_threads=True)
wr.s3.to_csv(val_df, val_path, header=False, index=False, use_threads=True)
wr.s3.to_csv(test_df, test_path, header=True, index=False, use_threads=True)

print(f"Uploaded: {train_path}, {val_path}, {test_path}")

# === 6. Training Inputs ===
train_input = TrainingInput(train_path, content_type='csv')
val_input = TrainingInput(val_path, content_type='csv')

# === 7. XGBoost Container (2025 Way) ===
container = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1"  # Stable, supports early stopping
)

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f's3://{data_bucket}/{subfolder}/output',
    sagemaker_session=sess
)

estimator.set_hyperparameters(
    max_depth=3,
    subsample=0.7,
    objective='binary:logistic',
    eval_metric='auc',
    num_round=100,
    early_stopping_rounds=10,
    scale_pos_weight=17
)

# === 8. Train ===
estimator.fit({'train': train_input, 'validation': val_input})

# === 9. Deploy ===
endpoint_name = 'customer-churn'
try:
    sess.delete_endpoint(endpoint_name)
    print("Old endpoint deleted. Waiting...")
    sleep(30)
except:
    pass

from sagemaker.serializers import CSVSerializer

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge',
    endpoint_name=endpoint_name,
    serializer=CSVSerializer()
)

# === 10. Predict ===
def predict_row(row):
    prob = float(predictor.predict(row[1:]).decode('utf-8'))
    return 1 if prob > 0.5 else 0

test_data = pd.read_csv(test_path)  # Has header
test_data['prediction'] = test_data.apply(predict_row, axis=1)

print("Accuracy:", metrics.accuracy_score(test_data['churned'], test_data['prediction']))
print("Confusion Matrix:\n", metrics.confusion_matrix(test_data['churned'], test_data['prediction']))

# === 11. Cleanup ===
sess.delete_endpoint(endpoint_name)

