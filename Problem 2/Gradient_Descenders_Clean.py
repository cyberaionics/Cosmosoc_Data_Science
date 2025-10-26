import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import random 


train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
if test_df is None:
    print("File not found. Please check the file path.")
if train_df is None:
    print("File not found. Please check the file path.")

figplt=plt.figure(figsize=(14,6))
sns.countplot(data=train_df, x='Age', hue='Transported')
plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
plt.xticks(rotation=90)
plt.title('Survival by Age')
plt.show()

figplt=plt.figure(figsize=(14,6))
sns.countplot(data=train_df, x='VIP', hue='Transported')
plt.title('Distribution of VIP Status by Transported')
plt.show()


train_df['VIP'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Passenger Class Distribution')
plt.show()

figplt=plt.figure(figsize=(14,6))
sns.countplot(data=train_df, x='HomePlanet', hue='Transported')
plt.title('Distribution of HomePlanet by Transported')
plt.show()

figplt=plt.figure(figsize=(14,6))
sns.countplot(data=train_df, x='Destination', hue='Transported')  
plt.title('Distribution of Destination by Transported')
plt.show()

combined_df = pd.concat([train_df, test_df])
combined_df = combined_df.sort_values(by='PassengerId').reset_index(drop=True)


print(combined_df.info())
print(combined_df.describe())

print("Missing values in each column:")
print(combined_df.isnull().sum())
print("Duplicate rows in the dataset:")
print(combined_df.duplicated().sum())

median_age=combined_df['Age'].median()
combined_df['Age'].fillna(median_age,inplace=True)

median_cryosleep=combined_df['CryoSleep'].mode()
combined_df['CryoSleep'].fillna(median_cryosleep,inplace=True)

combined_df.loc[combined_df['VIP'].isnull(), 'VIP'] = random.choices(
    [True,False],
    weights=[0.02,0.98],
    k=combined_df['VIP'].isnull().sum()
)

def rfssv(cell_value):
    if pd.isnull(cell_value):
        return 0
    else:
        return cell_value
col_to_update = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
combined_df[col_to_update] = combined_df[col_to_update].applymap(rfssv)




combined_df['Surname'] = combined_df['Name'].str.split().str[-1]

surname_cabin_map = combined_df.groupby('Surname')['Cabin'].apply(
       lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
 ).to_dict()
combined_df[['Cabin','HomePlanet','Destination']] = combined_df.apply(
        lambda row: surname_cabin_map.get(row['Surname'], None) 
        if pd.isna(row[['Cabin','HomePlanet','Destination']]).any() and not pd.isna(row['Surname']) 
        and surname_cabin_map.get(row['Surname'], None) is not None 
        else row[['Cabin','HomePlanet','Destination']], 
        axis=1
    )


combined_df.drop(['Surname', 'Transported'], axis=1, inplace=True)

combined_df['Destination'].loc[combined_df['Destination'].isnull()] = random.choices(
    ['TRAPPIST-1e','PSO J318.5-22','55 Cancri e'],
    weights=[0.70,0.21,0.09],
    k=combined_df['Destination'].isnull().sum()
)
combined_df['HomePlanet'].loc[combined_df['HomePlanet'].isnull()] = random.choices(
    ['Earth','Europa','Mars'],
    weights=[0.54,0.25,0.21],
    k=combined_df['HomePlanet'].isnull().sum()
)

combined_df['Cabin'].loc[combined_df['Cabin'].isnull()] = random.choices(
    combined_df['Cabin'].dropna().unique().tolist(),
    k=combined_df['Cabin'].isnull().sum()
)

combined_df['Name'].fillna('Unknown', inplace=True)

print(combined_df.isnull().sum())

combined_df .to_csv('cleaned_dataset.csv', index=False)