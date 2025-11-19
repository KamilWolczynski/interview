# AWS ML Architecture Review: 2020 Code Modernization

## Executive Summary

This document provides an AWS ML Architect's perspective on modernizing a 2020 Random Cut Forest anomaly detection implementation. The original code uses deprecated patterns and misses significant AWS advancements from the past 4+ years.

---

## Critical Issues & Modernization Priorities

### 🔴 **HIGH PRIORITY: Deprecated & Removed Components**

#### 1. **SageMaker SDK v1 → v2 Migration**
**Problem:** Code uses SageMaker Python SDK v1 patterns (deprecated in 2020, removed in 2023)

```python
# ❌ OLD (2020) - DEPRECATED
from sagemaker import RandomCutForest
rcf = RandomCutForest(role=role, ...)
rcf.fit(rcf.record_set(train_df_no_result.values))
```

**Solution:** Use SageMaker SDK v2 with built-in algorithms via Estimator

```python
# ✅ NEW (2024+)
from sagemaker.estimator import Estimator

rcf = Estimator(
    image_uri=sagemaker.image_uris.retrieve("randomcutforest", region),
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',  # Updated instance family
    output_path=f's3://{data_bucket}/{subfolder}/output',
    sagemaker_session=session
)

# Set hyperparameters explicitly
rcf.set_hyperparameters(
    num_samples_per_tree=100,
    num_trees=50,
    feature_dim=train_df_no_result.shape[1]
)

# Use TrainingInput for better control
from sagemaker.inputs import TrainingInput
train_input = TrainingInput(
    s3_data=train_s3_path,
    content_type='text/csv;label_size=0'
)
rcf.fit(train_input)
```

#### 2. **Instance Type Obsolescence**
**Problem:** `ml.m4.xlarge` and `ml.t2.medium` are 2-3 generations old

**Solution:**
- Training: `ml.m4.xlarge` → `ml.m5.xlarge` or `ml.m6i.xlarge` (20-40% better price/performance)
- Inference: `ml.t2.medium` → `ml.t3.medium` or **Serverless Inference** (cost-effective for variable traffic)

```python
# ✅ Serverless Inference (2024 best practice)
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5
)

predictor = rcf.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name=endpoint_name
)
```

#### 3. **Deprecated Serializers**
**Problem:** `csv_serializer` and `json_deserializer` removed in SDK v2

```python
# ❌ OLD - REMOVED
from sagemaker.predictor import csv_serializer, json_deserializer
rcf_endpoint.serializer = csv_serializer
rcf_endpoint.deserializer = json_deserializer
```

**Solution:**
```python
# ✅ NEW
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()
```

---

### 🟡 **MEDIUM PRIORITY: Architecture & Best Practices**

#### 4. **Data Access: s3fs → AWS Data Wrangler**
**Problem:** `s3fs` is generic; doesn't leverage AWS-native optimizations

```python
# ❌ OLD
import s3fs
s3 = s3fs.S3FileSystem(anon=False)
df = pd.read_csv(f's3://{data_bucket}/{subfolder}/{dataset}')
```

**Solution:**
```python
# ✅ NEW - AWS Data Wrangler (awswrangler)
import awswrangler as wr

# Faster, handles partitions, integrates with Glue Catalog
df = wr.s3.read_csv(
    path=f's3://{data_bucket}/{subfolder}/{dataset}',
    boto3_session=boto3.Session()
)

# Or use Glue Catalog for governed data access
df = wr.athena.read_sql_query(
    sql="SELECT * FROM activities WHERE error = true",
    database="ml_database"
)
```

**Benefits:**
- 3-10x faster for large datasets (parallel reads)
- Native Parquet/ORC support (columnar formats)
- Automatic schema inference and Glue Catalog integration

#### 5. **Feature Engineering: Manual One-Hot → SageMaker Processing**
**Problem:** Feature engineering in notebook = not reproducible in production

```python
# ❌ OLD - Notebook-only transformation
encoded_df = pd.get_dummies(df, columns=['Matter Type','Resource','Activity'])
```

**Solution:** Use SageMaker Processing with SKLearn for reproducible pipelines

```python
# ✅ NEW - Reusable preprocessing script
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

sklearn_processor = SKLearnProcessor(
    framework_version='1.2-1',
    role=role,
    instance_type='ml.m5.xlarge',
    instance_count=1
)

sklearn_processor.run(
    code='preprocessing.py',  # Separate script
    inputs=[ProcessingInput(
        source=f's3://{data_bucket}/{subfolder}/{dataset}',
        destination='/opt/ml/processing/input'
    )],
    outputs=[
        ProcessingOutput(output_name='train', source='/opt/ml/processing/train'),
        ProcessingOutput(output_name='validation', source='/opt/ml/processing/val')
    ]
)
```

**preprocessing.py:**
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv('/opt/ml/processing/input/activities.csv')

# Fit encoder (save for inference)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['Matter Type', 'Resource', 'Activity']])

# Save encoder for inference
joblib.dump(encoder, '/opt/ml/processing/model/encoder.pkl')

# Save processed data
train_df.to_csv('/opt/ml/processing/train/train.csv', index=False)
val_df.to_csv('/opt/ml/processing/val/validation.csv', index=False)
```

#### 6. **Model Evaluation: Manual Threshold → SageMaker Model Monitor**
**Problem:** Manual threshold calculation doesn't scale or monitor drift

```python
# ❌ OLD - One-time manual evaluation
score_cutoff = results_df[results_df['Error'] == True]['score'].median()
results_df['Prediction'] = results_df['score'] > score_cutoff
```

**Solution:** Implement automated monitoring

```python
# ✅ NEW - Continuous monitoring
from sagemaker.model_monitor import DataCaptureConfig, ModelQualityMonitor

# Enable data capture
data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f's3://{data_bucket}/monitoring/data-capture'
)

predictor = rcf.deploy(
    data_capture_config=data_capture_config,
    ...
)

# Set up quality monitoring
model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    max_runtime_in_seconds=1800
)

# Schedule monitoring jobs
model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name='rcf-quality-monitor',
    endpoint_input=predictor.endpoint_name,
    ground_truth_input=f's3://{data_bucket}/ground-truth/',
    problem_type='BinaryClassification',
    schedule_cron_expression='cron(0 * * * ? *)'  # Hourly
)
```

#### 7. **Endpoint Management: Manual Deletion → Lifecycle Management**
**Problem:** Manual endpoint cleanup is error-prone and costly

```python
# ❌ OLD - Manual try/except cleanup
try:
    session.delete_endpoint(sagemaker.predictor.RealTimePredictor(endpoint=endpoint_name).endpoint)
except:
    pass
```

**Solution:** Use auto-scaling and async inference for cost optimization

```python
# ✅ NEW - Auto-scaling for real-time
import boto3

client = boto3.client('application-autoscaling')

# Register scalable target
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=0,  # Scale to zero when idle
    MaxCapacity=3
)

# Target tracking policy
client.put_scaling_policy(
    PolicyName='SageMakerEndpointInvocationScalingPolicy',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```

**Or use Async Inference for batch-like workloads:**
```python
# ✅ Async Inference (2024 pattern)
from sagemaker.async_inference import AsyncInferenceConfig

async_config = AsyncInferenceConfig(
    output_path=f's3://{data_bucket}/async-inference/output',
    max_concurrent_invocations_per_instance=4
)

predictor = rcf.deploy(
    async_inference_config=async_config,
    instance_type='ml.m5.xlarge',
    initial_instance_count=1
)
```

---

### 🟢 **LOW PRIORITY: Enhancements & Future-Proofing**

#### 8. **MLOps: Notebook → SageMaker Pipelines**
**Problem:** Notebook code isn't production-ready or CI/CD compatible

**Solution:** Migrate to SageMaker Pipelines for orchestration

```python
# ✅ NEW - Production ML Pipeline
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString

# Define parameters
data_bucket_param = ParameterString(name="DataBucket", default_value=data_bucket)
instance_type_param = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")

# Step 1: Preprocessing
step_process = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[...],
    outputs=[...]
)

# Step 2: Training
step_train = TrainingStep(
    name="TrainRCFModel",
    estimator=rcf,
    inputs={
        'train': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs['train'].S3Output.S3Uri
        )
    }
)

# Step 3: Model registration
step_register = RegisterModel(
    name="RegisterRCFModel",
    estimator=rcf,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.t3.medium", "ml.m5.large"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="rcf-anomaly-detection"
)

# Create pipeline
pipeline = Pipeline(
    name="RCFAnomalyDetectionPipeline",
    parameters=[data_bucket_param, instance_type_param],
    steps=[step_process, step_train, step_register]
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

#### 9. **Experiment Tracking: Manual → SageMaker Experiments**
**Problem:** No tracking of hyperparameters, metrics, or model versions

**Solution:**
```python
# ✅ NEW - Automatic experiment tracking
from sagemaker.experiments import Run

with Run(
    experiment_name="rcf-anomaly-detection",
    run_name=f"rcf-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    sagemaker_session=session
) as run:
    # Log parameters
    run.log_parameters({
        "num_trees": 50,
        "num_samples_per_tree": 100,
        "instance_type": "ml.m5.xlarge"
    })
    
    # Train model
    rcf.fit(train_input)
    
    # Log metrics
    run.log_metric(name="validation_auc", value=0.87)
    run.log_metric(name="threshold", value=score_cutoff)
```

#### 10. **Security & Governance**

**Add VPC Configuration:**
```python
# ✅ Network isolation
from sagemaker.network import NetworkConfig

network_config = NetworkConfig(
    enable_network_isolation=False,
    security_group_ids=['sg-xxxxx'],
    subnets=['subnet-xxxxx', 'subnet-yyyyy']
)

rcf = Estimator(
    ...,
    subnets=network_config.subnets,
    security_group_ids=network_config.security_group_ids
)
```

**Add Encryption:**
```python
# ✅ Encryption at rest and in transit
rcf = Estimator(
    ...,
    volume_kms_key='arn:aws:kms:region:account:key/xxxxx',
    output_kms_key='arn:aws:kms:region:account:key/xxxxx',
    enable_network_isolation=True
)
```

**Add IAM Least Privilege:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::mlforbusiness/ch05/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:CreateEndpoint"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "us-east-1"
        }
      }
    }
  ]
}
```

---

## Cost Optimization Recommendations

### Current Architecture (2020)
- **Training:** ml.m4.xlarge @ $0.24/hr × 1 instance
- **Inference:** ml.t2.medium @ $0.065/hr × 24/7 = **$47/month** (always on)

### Optimized Architecture (2024)
1. **Training:** ml.m5.xlarge @ $0.192/hr (20% cheaper, 40% faster)
2. **Inference Options:**
   - **Serverless:** $0.20 per 1M requests + $0.0000133/ms compute = **~$5-15/month** (variable load)
   - **Async:** ml.m5.large @ $0.096/hr with auto-scaling to 0 = **$10-30/month** (batch)
   - **Real-time with auto-scaling:** ml.t3.medium @ $0.052/hr with scale-to-zero = **$15-35/month**

**Potential Savings: 60-90% on inference costs**

---

## Migration Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- [ ] Update SageMaker SDK to v2 (breaking changes)
- [ ] Replace s3fs with awswrangler
- [ ] Update instance types (m4→m5, t2→t3)
- [ ] Fix deprecated serializers

### Phase 2: Architecture Improvements (2-4 weeks)
- [ ] Extract preprocessing to SageMaker Processing job
- [ ] Implement Serverless or Async Inference
- [ ] Add SageMaker Experiments tracking
- [ ] Set up Model Monitor for drift detection

### Phase 3: Production Hardening (4-8 weeks)
- [ ] Migrate to SageMaker Pipelines
- [ ] Add VPC and encryption
- [ ] Implement auto-scaling policies
- [ ] Set up CI/CD with CodePipeline
- [ ] Add CloudWatch dashboards and alarms

### Phase 4: Advanced Features (8+ weeks)
- [ ] Multi-model endpoints (if multiple models)
- [ ] A/B testing with production variants
- [ ] Shadow mode deployment
- [ ] Feature Store integration for feature reuse

---

## Code Comparison: Before & After

### Before (2020)
```python
# Monolithic notebook with manual steps
data_bucket = 'mlforbusiness'
df = pd.read_csv(f's3://{data_bucket}/ch05/activities.csv')
encoded_df = pd.get_dummies(df, columns=['Matter Type','Resource','Activity'])
train_df, val_df = train_test_split(encoded_df, test_size=0.2)

rcf = RandomCutForest(role=role, train_instance_type='ml.m4.xlarge', ...)
rcf.fit(rcf.record_set(train_df_no_result.values))
rcf_endpoint = rcf.deploy(instance_type='ml.t2.medium', endpoint_name='suspicious-lines')

results = rcf_endpoint.predict(val_df_no_result.values)
score_cutoff = results_df[results_df['Error'] == True]['score'].median()
```

### After (2024)
```python
# Modular, production-ready pipeline
import awswrangler as wr
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.serverless import ServerlessInferenceConfig

# Step 1: Data ingestion (reusable)
df = wr.s3.read_csv(f's3://{data_bucket}/ch05/activities.csv')

# Step 2: Preprocessing (SageMaker Processing)
preprocessing_step = ProcessingStep(
    name="Preprocess",
    processor=sklearn_processor,
    code='preprocessing.py'
)

# Step 3: Training (with experiments)
with Run(experiment_name="rcf-anomaly") as run:
    rcf = Estimator(
        image_uri=sagemaker.image_uris.retrieve("randomcutforest", region),
        instance_type='ml.m5.xlarge',
        ...
    )
    rcf.fit(TrainingInput(preprocessing_step.properties.ProcessingOutputConfig...))

# Step 4: Serverless deployment
predictor = rcf.deploy(
    serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=2048),
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

# Step 5: Automated monitoring
model_monitor.create_monitoring_schedule(...)
```

---

## Key Takeaways for ML Architects

1. **SDK v2 is mandatory** - v1 is completely removed from PyPI
2. **Serverless inference** is the default for variable workloads (60-90% cost savings)
3. **SageMaker Pipelines** are essential for production ML (not optional)
4. **awswrangler** is the standard for AWS data access (not s3fs/boto3)
5. **Model Monitor** prevents silent model degradation in production
6. **Experiment tracking** is built-in (no need for MLflow/Weights&Biases)
7. **Instance families** matter - m5/m6i are 20-40% better price/performance than m4

---

## Additional Resources

- [SageMaker SDK v2 Migration Guide](https://sagemaker.readthedocs.io/en/stable/v2.html)
- [AWS Data Wrangler Documentation](https://aws-sdk-pandas.readthedocs.io/)
- [SageMaker Pipelines Workshop](https://catalog.workshops.aws/sagemaker-pipelines/)
- [Random Cut Forest Algorithm Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html)
- [Serverless Inference Pricing](https://aws.amazon.com/sagemaker/pricing/)

---

## Conclusion

This 2020 code requires significant modernization to meet 2024 AWS ML standards. The migration is **not optional** due to deprecated SDK components, but presents an opportunity to:

- **Reduce costs by 60-90%** (serverless inference)
- **Improve reliability** (pipelines, monitoring)
- **Enable CI/CD** (infrastructure as code)
- **Meet compliance requirements** (VPC, encryption, IAM)

**Recommended approach:** Start with Phase 1 (SDK updates) immediately, then prioritize Phase 2 (serverless inference) for cost savings. Phases 3-4 can be implemented incrementally based on production requirements.
