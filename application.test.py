import requests
import pandas as pd
import time
import matplotlib.pyplot as plt

# Define the URL of your deployed application
url = "http://ece444-pra5-env.eba-wwf7vy37.us-east-1.elasticbeanstalk.com/predict"  # Replace with your actual URL

# Define the test cases
test_cases = [
    {"text": "Aliens have landed on Earth and are taking over the government."},  # Fake news
    {"text": "A new study claims that drinking coffee cures cancer."},  # Fake news
    {"text": "The stock market reached an all-time high today, reflecting economic growth."},  # Real news
    {"text": "Local community center is offering free meals to those in need this winter."}  # Real news
]

# Prepare to collect timestamps
results = []

# Run each test case 100 times
for case in test_cases:
    for _ in range(100):
        start_time = time.time()
        response = requests.post(url, json=case)
        end_time = time.time()
        results.append({"text": case["text"], "response": response.json(), "timestamp": end_time - start_time})

# Create a DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv("api_performance_results.csv", index=False)

# Optional: Calculate average latency
average_performance = df.groupby("text")["timestamp"].mean()
print(average_performance)

# Generate a boxplot
plt.figure(figsize=(10, 6))
df.boxplot(column='timestamp', by='text', grid=False)
plt.title('API Response Time by Test Case')
plt.suptitle('')
plt.xlabel('Test Case')
plt.ylabel('Response Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

