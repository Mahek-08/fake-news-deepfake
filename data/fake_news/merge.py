
import pandas as pd
import os

fake_path = "data/fake_news/Fake.csv"
true_path = "data/fake_news/True.csv"
output_path = "data/fake_news/train.csv"

df_fake = pd.read_csv(fake_path)
df_true = pd.read_csv(true_path)

df_fake["label"] = "FAKE"
df_true["label"] = "REAL"

df_fake = df_fake[["title", "text", "label"]]
df_true = df_true[["title", "text", "label"]]

df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ train.csv created at {output_path} with {len(df)} rows")
