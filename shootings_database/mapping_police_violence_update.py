import pandas as pd
import os, sys
from datetime import datetime
file_loc = sys.path[0]
sys.path.append(os.path.dirname(file_loc))  # Add current file directory to path
from shootings_database.utils import address_parser
from shootings_database.utils  import agencyutils
from shootings_database.utils  import ois_matching
from shootings_database.utils  import opd_logger
import openpolicedata as opd
import logging

# Script for identifying police killing in OpenPoliceData (OPD) data not in MPV database.
# They are logged to a datestamped file. If the script is run multiple times, cases will 
# only be logged if they have not previously been logged.

############## SETUP PARAMETERS #########################################
# File locations 
db_filename = "https://mappingpoliceviolence.us/s/MPVDatasetDownload.xlsx"  # Link on homepage of https://mappingpoliceviolence.us/
output_dir = os.path.join(file_loc, r"MappingPoliceViolence", "Updates") # Where to output cases found

min_date = None   # Cases will be ignored before this date. If None, min_date will be set to the oldest date in MPV's data

############## OTHER CONFIGURATION PARAMETERS ###########################

# Names of columns that are not automatically identified
mpv_addr_col = "Street Address of Incident"
mpv_state_col = 'State'
mpv_agency_col = "Agency responsible for death"

# Parameters that affect which cases are logged
include_unknown_fatal = False  # Whether to include cases where there was a shooting but it is unknown if it was fatal 
log_demo_diffs = False  # Whether to log cases where a likely match between MPV and OPD cases was found but listed race or gender differs
log_age_diffs = False  # Whether to log cases where a likely match between MPV and OPD cases was found but age differs
keep_self_inflicted = False   # Whether to keep cases that are marked self-inflicted in the data

# Logging and restarting parameters
istart = 1  # 1-based index (same as log statements) to start at in OPD datasets. Can be useful for restarting. Set to 1 to start from beginning.
logging_level = logging.INFO  # Logging level. Change to DEBUG for some additional messaging.
unexpected_conditions = 'ignore'   # If 'error', an error will be thrown when a condition occurs that was not previously identified in testing.

# There are sometimes demographic differences between MPV and other datasets for the same case
# If a perfect demographics match is not found, an attempt can be made to allow differences in race and gender values
# when trying to find a case with matching demographics. The below cases have been identified  as differing in some cases.
# The pairs below will be considered equivalent when allowed_replacements is used
allowed_replacements = {'race':[["HISPANIC/LATINO","INDIGENOUS"],["HISPANIC/LATINO","WHITE"],["HISPANIC/LATINO","BLACK"],
                                ['ASIAN','ASIAN/PACIFIC ISLANDER']],
                        'gender':[['TRANSGENDER','MALE'],['TRANSGENDER','FEMALE']]}
    
####################################################################

if not os.path.exists(output_dir):
    print(f"Creating output directory {output_dir}")
    os.mkdir(output_dir)

logger = ois_matching.get_logger(logging_level)

# Load MPV database and convert to OPD table so that standardization can be applied and some 
# column names and terms for race and gender can be standardized
logger.info(f"Loading data from: {db_filename}")
mpv_raw = pd.read_excel(db_filename)
mpv_table = opd.data.Table({"SourceName":"Mapping Police Violence", 
                      "State":opd.defs.MULTI, 
                      "TableType":opd.TableType.SHOOTINGS,
                      'agency_field':mpv_agency_col}, 
                     mpv_raw,
                     opd.defs.MULTI)
mpv_table.standardize()
df_mpv = mpv_table.table  # Retrieve pandas DataFrame from Table class

# Standard column names for all datasets that have these columns
date_col = opd.Column.DATE
agency_col = opd.Column.AGENCY
role_col = opd.Column.SUBJECT_OR_OFFICER
zip_col = opd.Column.ZIP_CODE

# Standard demographic column names for MPV
mpv_race_col = ois_matching.get_race_col(df_mpv)
mpv_gender_col = ois_matching.get_gender_col(df_mpv)
mpv_age_col = ois_matching.get_age_col(df_mpv)

min_date = pd.to_datetime(min_date) if min_date else df_mpv[date_col].min()

# Get a list of officer-involved shootings and use of force datasets in OPD
tables_to_use = [opd.TableType.SHOOTINGS, opd.TableType.SHOOTINGS_INCIDENTS,
                 opd.TableType.USE_OF_FORCE, opd.TableType.USE_OF_FORCE_INCIDENTS]
opd_datasets = []
for t in tables_to_use:
    opd_datasets.append(opd.datasets.query(table_type=t))
opd_datasets = pd.concat(opd_datasets, ignore_index=True)
logger.info(f"{len(opd_datasets)} officer-involved shootings or use of force datasets found in OPD")

for k, row_dataset in opd_datasets.iloc[max(1,istart)-1:].iterrows():  # Loop over OPD OIS datasets
    logger.info(f'Running {k+1} of {len(opd_datasets)}: {row_dataset["SourceName"]} {row_dataset["TableType"]} {row_dataset["Year"] if row_dataset["Year"]!="MULTIPLE" else ""}')

    # Load this OPD dataset
    src = opd.Source(row_dataset["SourceName"], state=row_dataset["State"])    # Create source for agency
    try:
        # url_contains is typically not needed but is useful when looping over many datasets 
        # to handle cases where multiple datasets match the TableType and Year
        opd_table = src.load(row_dataset['TableType'], row_dataset['Year'], url_contains=row_dataset['URL'])  # Load data
    except:
        if unexpected_conditions=='error':
            raise ValueError(f"{row_dataset['TableType']} dataset for the year {row_dataset['Year']} not available for {row_dataset['SourceName']}, {row_dataset['State']}")
        else:
            # Website where the data exists is likely down
            print(f"{row_dataset['TableType']} dataset for the year {row_dataset['Year']} not available for {row_dataset['SourceName']}, {row_dataset['State']}. "+
                  "The website may be temporarily unavailable.")
            continue
    opd_table.standardize(agg_race_cat=True)  # Standardize data
    opd_table.expand(mismatch='splitsingle')  # Expand cases where the info for multiple people are contained in the same row
    # Some tables contain incident information in 1 table and subject and/or officer information in other tables
    related_table, related_years = src.find_related_tables(opd_table.table_type, opd_table.year, sub_type='SUBJECTS')
    if related_table:
        t2 = src.load(related_table[0], related_years[0])
        t2.standardize(agg_race_cat=True)
        try:
            # Merge incident and subjects tables on their unique ID columns to create 1 row per subject
            opd_table = opd_table.merge(t2, std_id=True)
        except opd.exceptions.AutoMergeError as e:
            if len(e.args)>0 and e.args[0]=='Unable to automatically find ID that relates tables' and \
                row_dataset["SourceName"]=='Charlotte-Mecklenburg':
                # Dataset has no incident ID column. Latitude/longitude seems to work instead
                opd_table = opd_table.merge(t2, on=['Latitude','Longitude'])
            else:
                raise
        except:
            raise
    df_opd_all = opd_table.table

    # Get standardized demographics columns for OPD data
    opd_race_col = opd_table.get_race_col()
    opd_gender_col = opd_table.get_gender_col()
    opd_age_col = opd_table.get_age_col()

    df_opd_all, known_fatal, test_cols = ois_matching.clean_data(opd_table, df_opd_all, row_dataset['TableType'], min_date, 
                                                  include_unknown_fatal, keep_self_inflicted)

    if len(df_opd_all)==0:
        continue  # No data. Move to the next dataset

    # Find address or street column if it exists
    addr_col = address_parser.find_address_col(df_opd_all, error=unexpected_conditions)
    addr_col = addr_col[0] if len(addr_col)>0 else None

    # If dataset has multiple agencies, loop over them individually
    agency_names = df_opd_all[opd.Column.AGENCY].unique() if row_dataset['Agency']==opd.defs.MULTI else [row_dataset['AgencyFull']]
    for agency in agency_names:
        if row_dataset['Agency']==opd.defs.MULTI:
            df_opd = df_opd_all[df_opd_all[opd.Column.AGENCY]==agency].copy()
        else:
            df_opd = df_opd_all.copy()

        # Get the location (agency_partial) and type (police department, sheriff's office, etc.) from the full agency name
        agency_partial, agency_type = agencyutils.split(agency, row_dataset['State'], unknown_type=unexpected_conditions)
        # Only keep rows that might correspond to the current agency
        df_mpv_agency = agencyutils.filter_agency(agency, agency_partial, agency_type, row_dataset['State'], 
                                             df_mpv, agency_col, mpv_state_col, logger=logger, error=unexpected_conditions)
        
        # Match OPD cases to MPV cases starting with strictest match requirements and then
        # with progressively more relaxed requirements. There are frequent differences between
        # the datasets in OPD and MPV that need to be dealt with. 

        # args first requires a perfect demographics match then loosen demographics matching requirements
        # See ois_matching.check_for_match for definitions of the different methods for loosening 
        # demographics matching requirements
        args = [{}, {'max_age_diff':1,'check_race_only':True}, 
                {'allowed_replacements':allowed_replacements},
                {'inexact_age':True}, {'max_age_diff':5}, {'allow_race_diff':True},{'max_age_diff':20, 'zip_match':True},
                {'max_age_diff':10, 'zip_match':True, 'allowed_replacements':allowed_replacements}]
        subject_demo_correction = {}
        match_with_age_diff = {}
        mpv_matched = pd.Series(False, df_mpv_agency.index)
        for a in args:
            # First find cases that have the same date and then check demographics and possibly zip code. Remove matches.
            df_opd, mpv_matched, subject_demo_correction, match_with_age_diff = ois_matching.remove_matches_date_match_first(
                df_mpv_agency, df_opd, mpv_addr_col, addr_col, 
                mpv_matched, subject_demo_correction, match_with_age_diff, a, 
                test_cols, error=unexpected_conditions)
        
        # First find cases that have the same demographics and then check if date is close and street matches (if there is an address column).
        #  Remove matches.
        df_opd, mpv_matched = ois_matching.remove_matches_demographics_match_first(df_mpv_agency, df_opd, 
                                                                                   mpv_addr_col, addr_col, mpv_matched,
                                                                                   error=unexpected_conditions)

        if addr_col:
            # First find cases that have the same street and then check if date is close.  Remove matches.
            df_opd, mpv_matched, subject_demo_correction = ois_matching.remove_matches_street_match_first(df_mpv_agency, df_opd, mpv_addr_col, addr_col,
                                      mpv_matched, subject_demo_correction, error=unexpected_conditions)
            # Sometimes, MPV has an empty or different agency so check other agencies.
            # First find street match and then if date is close and demographics match. Remove matches.
            df_opd = ois_matching.remove_matches_agencymismatch(df_mpv, df_mpv_agency, df_opd, mpv_state_col, row_dataset['State'], 
                                                                'address', mpv_addr_col, addr_col,
                                                                error=unexpected_conditions)
        else:
            # Remove cases where the zip code matches and the date is close
            df_opd, mpv_matched, match_with_age_diff = ois_matching.remove_matches_close_date_match_zipcode(
                    df_mpv_agency, df_opd, mpv_matched, match_with_age_diff, 
                    allowed_replacements=allowed_replacements, error=unexpected_conditions)
            # Sometimes, MPV has an empty or different agency so check other agencies.
            # First find zip code match and then if date is close and demographics match. Remove matches.
            df_opd = ois_matching.remove_matches_agencymismatch(df_mpv, df_mpv_agency, df_opd, mpv_state_col, row_dataset['State'], 
                                                                'zip', error=unexpected_conditions)
            
        df_opd, mpv_matched = ois_matching.remove_name_matches(df_mpv_agency, df_opd, mpv_matched, error=unexpected_conditions)
                
        # Create a table with columns specific to this agency containing cases that may not already be in MPV
        name_col = opd.defs.columns.NAME_OFFICER_SUBJECT if opd.defs.columns.NAME_OFFICER_SUBJECT in df_opd else opd.defs.columns.NAME_SUBJECT
        df_save, keys = opd_logger.generate_agency_output_data(df_mpv_agency, df_opd, mpv_addr_col, addr_col, name_col,
                                log_demo_diffs, subject_demo_correction, log_age_diffs, match_with_age_diff, agency, known_fatal)
        
        if len(df_save)>0:
            # Save data specific to this source if it has not previously been saved
            source_basename = f"{row_dataset['SourceName']}_{row_dataset['State']}_{row_dataset['TableType']}_{row_dataset['Year']}"
            opd_logger.log(df_save, output_dir, source_basename, keys=keys, add_date=True, only_diffs=True)

            # Create a table with general columns applicable to all agencies that may not already be in MPV
            df_global = opd_logger.generate_general_output_data(df_save, addr_col, name_col)

            # CSV file containing all recommended updates with a limited set of columns
            global_basename = 'Potential_MPV_Updates_Global'
            keys = ["MPV ID", 'type', 'known_fatal', 'OPD Date','OPD Agency','OPD Race', 'OPD Gender','OPD Age','OPD Address']
            # Save general data to global file containing data for all OPD datasets
            opd_logger.log(df_global, output_dir, global_basename, keys=keys, add_date=True, only_diffs=True)
