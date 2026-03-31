import matplotlib.pyplot as plt
import openpolicedata as opd
import pandas as pd
import re

def add_publisher_deleted_datasets(df):
    df_deleted = pd.read_csv('https://raw.githubusercontent.com/openpolicedata/opd-data/refs/heads/main/datasets_deleted_by_publisher.csv')

    cols = [c for c in df_deleted.columns if c in df]
    df_deleted = df_deleted[cols]
    df_deleted['deleted'] = True
    df['deleted'] = False
    df.loc[df['supplying_entity']=='OpenPoliceData','deleted'] = True # We only host datasets that have been deleted
    df = pd.concat([df, df_deleted], ignore_index=True)

    return df

# Helper functions go in this cell
def remove_rows(remove_condition, all_datasets, create_debug_file, debug_filename=None):
    
    starting_all_datasets_count = all_datasets.shape[0]
    
    remove_datasets = all_datasets[remove_condition]
        
    # # Drop the identified rows
    all_datasets.drop(remove_datasets.index, inplace=True)
        
    # Verify the number of datasets removed
    assert len(remove_datasets) == (starting_all_datasets_count - all_datasets.shape[0]), \
        "Mismatch in the number of datasets removed"
        
    # Save removed datasets if a debug filename is provided
    if create_debug_file:
        remove_datasets.to_csv(debug_filename, index=True)
    
    return all_datasets

def validate_ois_datasets(df, up_to_date_dataset_year_max):
    keep = pd.Series(True, index=df.index)

    # Get Mapping Police Violence data. MPV is a pretty thorough tracking of police killings.
    # It does not include OIS but generally should be a good indicator whether an OIS dataset
    # has seized to be updated or there just haven't been any OIS
    src = opd.Source("Mapping Police Violence")
    t = src.load("OFFICER-INVOLVED SHOOTINGS", "MULTIPLE")
    t.standardize()

    df_mpv = t.table

    # Only keep killings via Gunshot. Others might not be included in OIS data
    df_mpv = df_mpv[df_mpv['Cause of death']=='Gunshot']
    
    for i in df[df['TableType']=='OFFICER-INVOLVED SHOOTINGS'].index:
        if df.loc[i,'MaxYear'] >= up_to_date_dataset_year_max:
            continue  # This dataset will be classified as up-to-date

        state_abbrev = opd.defs.states[df.loc[i, 'State']]
        df_mpv_coarse = df_mpv[df_mpv['State']==state_abbrev]  # Filter MPV for OPD state
        # Conservative filtering for OPD agency (only needs to contain partial agency name, not full one)
        df_mpv_coarse = df_mpv_coarse[df_mpv_coarse['AGENCY'].str.lower().str.contains(df.loc[i, 'SourceName'].lower())]

        num_coarse = len(df_mpv_coarse)

        # Clean up MPV agency names
        agencies = df_mpv_coarse['AGENCY'].apply(lambda x: re.sub(rf'\s\(?{state_abbrev}\)?\s', ' ', x, flags=re.IGNORECASE)).str.lower()
        # Filter MPV data for full agency name
        df_mpv_cur = df_mpv_coarse[agencies.str.contains(df.loc[i, 'AgencyFull'].lower())]

        if len(df_mpv_cur)==0:
            if num_coarse>0:
                raise ValueError(f"Unable to find any OIS in the MPV database for {df.loc[i, 'SourceName']}, {state_abbrev}")
        elif df_mpv_cur['DATE'].max().year>df.loc[i,'MaxYear']:
                continue  # MPV agrees that this dataset has stopped being updated

        # MPV data suggests that OPD dataset may be current despite most recent OIS being old
        keep.loc[i] = False
        print(f"{i}: Removing {df.loc[i, 'SourceName']}, {df.loc[i, 'State']}")

    df = remove_rows(remove_condition=~keep,
                     all_datasets=df, create_debug_file=False,
                     debug_filename='removed_ois_datasets.csv')

    return df


def get_stopped_data_stats(df, minimum_tabletype_counts_to_show, up_to_date_dataset_year_max):
    tabletype_analysis_df = df.copy()
    tabletype_counts = tabletype_analysis_df['TableType'].value_counts()

    # show only statistically significant table types
    total_tabletype_counts = len(tabletype_counts)
    tabletype_counts = tabletype_counts[tabletype_counts >= minimum_tabletype_counts_to_show]
    print(f"Table type counts >= {minimum_tabletype_counts_to_show} length is {len(tabletype_counts)}. The total number of table types is {total_tabletype_counts}")

    # filter out the table types that are not in the tabletype_counts
    tabletype_analysis_df = tabletype_analysis_df[tabletype_analysis_df['TableType'].isin(tabletype_counts.index)]

    print(f'Number of datasets from the {len(tabletype_counts)} tables is: {len(tabletype_analysis_df)}')

    stopped_datasets = tabletype_analysis_df[tabletype_analysis_df['MaxYear'] < up_to_date_dataset_year_max]

    return tabletype_counts, stopped_datasets


def stopped_data_by_table_type_plots(tabletype_counts, stopped_datasets, title_prepend=""):
    
    stopped_tabletype_counts = stopped_datasets['TableType'].value_counts()


    # compute a bar graph histogram of the number of datasets that are stopped by TableType
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    tabletype_counts.plot(kind='bar', ax=axes[0])
    axes[0].set_xlabel('Table Type')
    axes[0].set_ylabel('Number of Datasets')
    axes[0].set_title(title_prepend+'Number of Datasets by Table Type')
    axes[0].set_xticklabels(tabletype_counts.index, rotation=45, ha='right')

    stopped_tabletype_counts.plot(kind='bar', ax=axes[1])
    axes[1].set_xlabel('Table Type')
    axes[1].set_ylabel('Number of Datasets')
    axes[1].set_title(title_prepend+'Number of Stopped Datasets by Table Type')
    axes[1].set_xticklabels(stopped_tabletype_counts.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

    # Find which type has the highest ratio of stopped datasets
    percentage_of_stopped_datasets = 100*(stopped_tabletype_counts / tabletype_counts)
    percentage_of_stopped_datasets = percentage_of_stopped_datasets.fillna(0)

    # Create a bar plot of the ratio and sort the values from high to low
    percentage_of_stopped_datasets = percentage_of_stopped_datasets.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    ax = percentage_of_stopped_datasets.plot(kind='bar')
    ax.set_xticklabels(percentage_of_stopped_datasets.index, rotation=45, ha='right')
    plt.xlabel('Table Type')
    plt.ylabel('Percentage of Stopped Datasets')
    plt.title(title_prepend+'Percentage of Stopped Datasets by Table Type')
    for k in range(len(tabletype_counts)):
        plt.text(k - 0.25, percentage_of_stopped_datasets.iloc[k] + 0.3, 
                 f'{stopped_tabletype_counts.loc[percentage_of_stopped_datasets.index[k]]} / {tabletype_counts.loc[percentage_of_stopped_datasets.index[k]]}',
                 size='medium')
    plt.show()


def stopped_data_by_year(stopped_datasets, title_prepend=""):
    # Find the table type with highest stop percentage from previous analysis
    table_type = None #'OFFICER-INVOLVED SHOOTINGS'  # Enter None for all datasets or a table_type for specific table type
    if table_type:
        show_datasets = stopped_datasets[stopped_datasets['TableType']==table_type]
    else:  # Set to None to show all
        show_datasets = stopped_datasets
    year_counts = show_datasets['MaxYear'].value_counts().sort_index()
    year_counts.plot.bar(ylabel="# of Stopped Datasets", xlabel='Last Year of Data')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title(title_prepend+f"Stopped Datasets by Year for {table_type if table_type else 'All Datasets'}")
    plt.show()