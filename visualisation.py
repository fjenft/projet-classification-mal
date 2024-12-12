import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = '/home/onyxia/work/anime_data.csv' #make sure the path is correct!
df = pd.read_csv(csv_file_path)

# Correlation heatmap
numerical_features = ['num_list_users', 'num_episodes', 'mean', 'rank', 'popularity', 'num_scoring_users', 'start_year']
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
plt.savefig('/home/onyxia/work/heatmap.png')

# Popularity as a function of mean ratings
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['mean'], y=df['popularity'])
plt.title('Popularity vs. Mean Ratings')
plt.xlabel('Mean Ratings')
plt.ylabel('Popularity Rank')
plt.show()
plt.savefig('/home/onyxia/work/Popularity&mean_scores.png')

# Popularity as a function of number of episodes
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['num_episodes'], y=df['popularity'])
plt.title('Popularity vs. Number of Episodes')
plt.xlabel('Number of Episodes')
plt.ylabel('Popularity Rank')
plt.yscale('log')  # Log scale to better visualize bigger values
plt.show()
plt.savefig('/home/onyxia/work/Popularity&numbr_episodes.png')

# Popularity as a function of media type
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['media_type'], y=df['popularity'], order=df['media_type'].value_counts().index)
plt.title('Popularity vs. Media Type')
plt.xlabel('Media Type')
plt.ylabel('Popularity Rank')
plt.yscale('log')
plt.xticks(rotation=45)
plt.show()
plt.savefig('/home/onyxia/work/Popularity&type.png')