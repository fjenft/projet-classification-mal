import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = '/home/onyxia/work/anime_data.csv' #make sure the path is correct!
df = pd.read_csv(csv_file_path)

#printing first lines of the data
print(df.head())

#some statistics 
print(df.describe())

#type of each variable
print(df.dtypes)

#distributions of numerical variables
df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distributions of Numerical Variables")
plt.show()
plt.savefig('/home/onyxia/work/distributions.png')


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
plt.hexbin(x=df['mean'], y=df['popularity'], gridsize=50, cmap='Blues', mincnt=1)
plt.colorbar(label='Count')
plt.title('Popularity as a function of Mean Ratings')
plt.xlabel('Mean Ratings')
plt.ylabel('Popularity Rank')
plt.show()
plt.savefig('/home/onyxia/work/Popularity&mean_scores.png')

# Popularity as a function of number of users
plt.figure(figsize=(10, 6))
plt.scatter(df['num_list_users'], df['popularity'], alpha=0.6, s=10)
plt.xlim(0, df['num_list_users'].max())
plt.title('Popularity as a function of number of users')
plt.xlabel('num_list_users')
plt.ylabel('Popularity Rank')
plt.yscale('log')
plt.show()
plt.savefig('/home/onyxia/work/Popularity&numbr_users.png')

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
