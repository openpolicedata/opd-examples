# Overview
This folder contains code that compares police shootings and killings from the OpenPoliceData (OPD) Python library to those from a database such as [Mapping Police Violence](https://mappingpoliceviolence.org/), the [Washington Post Police Shootings Database](https://www.washingtonpost.com/graphics/investigations/police-shootings-database/), and [Fatal Encounters](https://fatalencounters.org/). The objective is to help these databases by finding historical cases available in an OPD dataset that *may* not be in the databases or to provide an additional method for finding recent shootings. 

We say *may* because there can be errors in the data or differences in how the data is recorded that cause a case to appear to not be in a database when in fact it is. However, we have implemented a lot of logic and checks to minimize errors. Matches are based on demographics, date, and location (address or zip code). We have checked a subset of the results by inspection and internet searches to ensure that the vast majority of cases appear to be new cases found in the OPD datasets.

Currently, it is implemented for Mapping Police Violence but can be easily modified for other databases.

# Installation
## Access the Code
[Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the OPD Examples repository with [Git](https://git-scm.com/):

```
> git clone https://github.com/openpolicedata/opd-examples.git
```

OR

Download the OPD Examples repository [here](https://codeload.github.com/openpolicedata/opd-examples/zip/refs/heads/main).

## Setup the Environment
Navigate to the opd-examples\shootings_database in a command prompt and run:
```
pip install -r requirements.txt
```

# Running the code
A script for comparing OPD to the shootings databases is located in [shootings_database\mapping_police_violence_update.py](https://github.com/openpolicedata/opd-examples/blob/main/shootings_database/mapping_police_violence_update.py). The code contains parameters at the top of the file that can be updated as needed. 

For the Mapping Police Violence database, the database must be downloaded to a local file from [Airtable](https://airtable.com/appzVzSeINK1S3EVR/shroOenW19l1m3w0H/tblxearKzw8W7ViN8). Update `csv_filename` in mapping_police_violence_update.py with the location of this file.

For each dataset where unique cases are found, a CSV output file is created for that OPD dataset with all values for each case from the dataset. A set of general information for each case from all OPD datasets will be stored in Potential_MPV_Updates_Global_{Current Date}.csv. All files are datestamped so if the code is run on different days, new files will be generated. If the code is run multiple times, only cases that have not previously been found will be stored. Therefore, it is possible to rerun the code at a future date to get updates, which will be stored in a new datestamped file.

# Help
Please email us [here](openpolicedata@gmail.com) if you have any questions, issues, or recommendations. We are happy to help get you started working with this code.