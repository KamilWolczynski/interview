I start from exploring categories to spot any anomalies.
What categories appear?
Are some categories very rare?
Are certain people or tasks unusual?

This is EDA – **exploratory data analysis**

Random Cut Forest:

Learns what normal invoice lines look like.

Measures how far a line is from normal.

Returns a score — high = suspicious.

I train it with: rcf.fit(...)

I delpoy: rcf.deploy(...)
Creates a real endpoint that can be called like an API.

That means:

Any system (ERP, invoicing tool, RPA) can hit the endpoint.

You send the invoice row → get a suspiciousness score back

results = rcf_endpoint.predict(val_df_no_result.values)

calculation
score_cutoff = median score of known errors

delete_endpoint()


### If something is extremely rare, what problem does the model have?


The model has very little historical data to understand what “normal” is for that category.

encoded_df = pd.get_dummies(df, columns=['Matter Type','Resource','Activity'])
Converts categorical text into numbers.

RCF (like most ML models) can only work with numbers, not strings.


One-hot encoding forces the model to learn relationships based on co-occurrence, not fake numeric proximity.

Corporate = 3
Litigation = 5
Financial  = 7

Litigation is halfway between Corporate and Financial.
Which is completely false in reality — categories have no numeric order.

### Why unsupervised learning?

Your answer was strong:

“There can be thousands of deviations.”

Supervised learning requires:

Labels: “good invoice line”, “bad invoice line”

Lots of examples of fraud or errors

But law firms don’t have this.
Most errors are never caught, and the ones caught are inconsistent.

Unsupervised learning is ideal when:

Errors are rare

Error types constantly change

You don’t know what future errors will look like

This is your key takeaway:

**Unsupervised anomaly detection detects patterns that haven’t even happened yet.**

### Why median cutoff?

Median is robust because it reflects the “typical abnormal score” without being pulled up by extreme cases.

### Why does the 2025 version look much longer?

encoded_df = pd.get_dummies(df, columns=['Matter Type','Resource','Activity'])
assumes everything else in the data is clean, numeric, balanced, small, and ready to train.

That’s almost never true in real projects.

In modern AWS ML engineering practice, we need to protect against:

Bad (non-numeric) columns

Missing values

Massive dimensionality after one-hot encoding

Columns that should not be used for anomaly scoring (e.g., IDs, text labels)

Unscaled numeric features that distort models

Supervised vs unsupervised differences

Production consistency (same transforms in prod as in training)

The longer code addresses all of those.

Let’s go line by line.