import pandas as pd

df = pd.read_csv('activity_dataset.csv')

#typos
# df['ideal_slot'] = df['ideal_slot'].replace('post_reset', 'post-reset')

# set average friction for empty friction values
def average_friction(df):
    # df['friction'] = df['friction'].fillna(df['friction'].mean())
    df['friction'] = df['friction'].apply( lambda x: "medium" if pd.isna(x) else x)


# if duration is <= 10 set low activity load
def assign_low_load_to_short_act(df):
    df['mental_load'] = df.apply( lambda row: 'low' if row['duration'] <= 10 else row['mental_load'], axis=1) 
    df['physical_load'] = df.apply( lambda row: 'low' if row['duration'] <= 10 else row['physical_load'], axis=1) 


#delete duplicates
def delete_duplicates(df):
    print(df['task_name'].value_counts()[df['task_name'].value_counts() > 1])
    df.drop_duplicates(subset='task_name', keep='first', inplace=True)


average_friction(df)
# assign_low_load_to_short_act(df)
delete_duplicates(df)

# push changes 
push_changes = lambda df : df.to_csv('activity_dataset.csv', index=False)

push_changes(df)


