import pandas as pd
import numpy as np
import os
import geopandas as gpa
import pathlib as pl
import shapely as sh
import fiona
# import libkml

fiona.supported_drivers["LIBKML"] = "raw"

# In[1]:
def CreateWidePositionsFrame(schedule_df, positions_df):
    #join the two input dataframes
    positions_df.reset_index('POSITION_INDEX', inplace = True)
    wide_df = schedule_df.join(positions_df)
    return wide_df

def MapOperationsColumns(df, source_type):
    #read in the common data fields table
    df_common_operations_fields = pd.read_excel('common_schedule_and_position_fields.xlsx')

    if source_type == 'FlightAware':
        remap = df_common_operations_fields[df_common_operations_fields['FlightAware field name'].notna()].set_index('FlightAware field name')[["Common field name"]].to_dict()
    elif source_type == 'AirNav':
        remap = df_common_operations_fields[df_common_operations_fields['AirNav field name'].notna()].set_index('AirNav field name')[["Common field name"]].to_dict()
        
    #return a dataframe with renamed columns
    
    return remap["Common field name"]

def IndexAirNavData(df, fn):
    #add FLIGHT_ID
    df["FLIGHT_ID"] = df.index    
    df.FLIGHT_ID = df.FLIGHT_ID.astype(str)
    df.FLIGHT_ID =  'AirNav_' + fn + '_' + df.FLIGHT_ID
    print('indexed ' + str(len(df.index)) + ' flights in file')
    return df

def PrepareAirNavData(airnav_input, fn, bb_filter, departure_airport, arrival_airport, write_normalized = True):
    # airnav = pd.DataFrame(airnav_input) #why do we copy this data?   for convenience / to make it a dataframe?
    # indexed_airnav = IndexAirNavData(airnav, fn).set_index('FLIGHT_ID')
    IndexAirNavData(airnav_input, fn).set_index('FLIGHT_ID', inplace = True)
    
    if departure_airport: 
        airnav_input = airnav_input[airnav_input.aptkoic == departure_airport]
    if arrival_airport:
        airnav_input = airnav_input[(airnav_input.aplngic == arrival_airport)] # & (airnav_input.aptkoic != arrival_airport)]

    print('processing ' + str(len(airnav_input)) + ' flights arriving and departing airport of interest')
        
    # airnav_positions = indexed_airnav['positions']
    airnav_positions = airnav_input['positions']    
    
    airnav_without_positions = airnav_input.drop(columns ='positions')    
    # airnav_without_positions = indexed_airnav.drop(columns ='positions')    
    # airnav_input.drop(columns ='positions', inplace = True)
    
    trajectories = pd.DataFrame()
    for flight_id, trajectory in airnav_positions.items():
        flight_trajectory = pd.DataFrame(trajectory) #why do we copy this data? for convenience / to make it a dataframe?
        flight_trajectory["FLIGHT_ID"] = flight_id
        flight_trajectory['POSITION_INDEX'] = flight_trajectory.index
        flight_trajectory.set_index(['FLIGHT_ID','POSITION_INDEX'], inplace = True)
        # trajectory["FLIGHT_ID"] = flight_id
        # trajectory['POSITION_INDEX'] = flight_trajectory.index
        # trajectory.set_index(['FLIGHT_ID','POSITION_INDEX'], inplace = True)
        ft = flight_trajectory
        # ft = trajectory
        ft.lat = ft.lat.astype('float')
        ft.lon = ft.lon.astype('float')
        if bb_filter:
            ft = ft[(ft.lat >= bb_filter['lat1']) & (ft.lat <= bb_filter['lat2']) & (ft.lon >= bb_filter['lon1']) & (ft.lon <= bb_filter['lon2'])]
        trajectories = pd.concat([ft, trajectories])
        print('Processed trajectory for flight' + flight_id)        

    print('Number of AirNav trajectory points: ', len(trajectories))
    
    wide_airnav = CreateWidePositionsFrame(airnav_without_positions, trajectories)
    output_dir_and_file_prefix = os.path.join(os.curdir, 'Output_arrivals/' + 'Modified_AirNav' + fn)
    if write_normalized:
        augmented_airnav = AugmentPositionData(trajectories.reset_index(), 'AirNav' )
        sorted_schedule_columns = sorted(airnav_without_positions.columns.to_list())
        sorted_position_columns = sorted(augmented_airnav.columns.to_list())
        if 'depchec' in sorted_schedule_columns:
            sorted_schedule_columns.remove('depchec') 
        
        airnav_without_positions[sorted_schedule_columns].to_csv(output_dir_and_file_prefix + '_schedule.csv', index = True)#, columns = sorted_schedule_columns)
        augmented_airnav[sorted_position_columns].to_csv(output_dir_and_file_prefix + '_trajectory.csv', index = True)#, columns = sorted_position_columns)
    else:
        augmented_airnav = AugmentPositionData(wide_airnav.reset_index(), 'AirNav' )
        return augmented_airnav

def AugmentPositionData(trajectories, type):
    #Brief check to make sure the type was input correctly
    if type not in ['AirNav', 'FlightAware']:
        print('The ModifyFunction input type must be either \'AirNav\' or \'FlightAware\'')
        quit()

    #Convert the time into a datetime object
    trajectories['Time (UTC)'] = pd.to_datetime(trajectories['svd'])    
    
    #Add a subindex that counts the datapoints within each unique flight
    if 'POSITION_INDEX' not in trajectories.keys():
        #Sort the data so it's all in order properly, by both time and flight ID
        trajectories = trajectories.sort_values(by = ['FLIGHT_ID', 'Time (UTC)'])
        trajectories = trajectories.set_index(trajectories.groupby('FLIGHT_ID').cumcount(), append = True)
        trajectories['POSITION_INDEX'] = trajectories.index.to_flat_index().map(lambda x: x[1])    

    trajectories.set_index('FLIGHT_ID', inplace = True)

    #perform group by on FLIGHT_ID in order to do flight specific determinations
    grouped = trajectories.groupby(trajectories.index.names)
    
    #prepare to access a flight group / position index combination 
    trajectories["max_position_index"] = grouped['POSITION_INDEX'].apply(lambda x: x.max())
    trajectories["is_max_position"] = trajectories.POSITION_INDEX == trajectories.max_position_index
    trajectories.set_index('POSITION_INDEX', append = True, inplace = True)
    
    #Use GeoPandas to convert the latitude and longitude to a geometric point for the current and next data
    trajectories = gpa.GeoDataFrame(trajectories)
    trajectories['POSITION_POINT'] = gpa.points_from_xy(trajectories['lon'], trajectories['lat'], crs = 4326)
    if 'aptkolo' in trajectories.columns.to_list():
        trajectories['TAKEOFF_APT_REF_PT'] = gpa.points_from_xy(trajectories['aptkolo'], trajectories['aptkola'], crs = 4326)
        trajectories['Line to airport'] = trajectories.apply(lambda x: sh.geometry.LineString([(x.lat, x.lon), (x["aptkola"], x["aptkolo"])]), axis=1)
    
    #find info about next point shifting index down by one
    trajectories['Next Latitude'] = trajectories['lat'].shift(periods = -1).astype('float')
    trajectories['Next Longitude'] = trajectories['lon'].shift(periods = -1).astype('float')
    trajectories['Duration (Seconds)'] = (trajectories['Time (UTC)'].shift(periods = -1) - trajectories['Time (UTC)']).dt.total_seconds()
    trajectories['SEGMENT'] = trajectories.apply(lambda x: sh.geometry.LineString([(x.lat, x.lon), (x["Next Latitude"], x["Next Longitude"])]), axis=1)
    trajectories['NEXT_POSITION_POINT'] = gpa.points_from_xy(trajectories['Next Longitude'], trajectories['Next Latitude'], crs = 4326)
    trajectories['GROUND_TRACK_LENGTH'] = np.nan

    #Ensure that the last point of a specific flight doesn't have the next flight's data copied over
    trajectories.loc[trajectories.is_max_position == True,'Next Latitude'] = np.nan
    trajectories.loc[trajectories.is_max_position == True, 'Next Longitude'] = np.nan
    trajectories.loc[trajectories.is_max_position == True, 'Duration (Seconds)'] = np.nan        
    trajectories.loc[trajectories.is_max_position == True, 'GROUND_TRACK_LENGTH'] = np.nan         
    trajectories.loc[trajectories.is_max_position == True, 'NEXT_POSITION_POINT'] = np.nan
        
    # ToDo: create geospatially enabled line segment and test its length compared to the calculated distance between points
    # #Geopandas requires you to loop through every row individually when making lines.... which is very slow and inefficient
    # trajectories['GROUND_TRACK'] = None
    # trajectories['GROUND_TRACK'] = trajectories['GROUND_TRACK'].astype(object)
    # for i in list(trajectories.index):
    #     trajectories.loc[i, 'GROUND_TRACK'] = sh.geometry.LineString([trajectories.loc[i, 'POSITION_POINT'], trajectories.loc[i, 'NEXT_POSITION_POINT']])
    
    return trajectories

def filterWithPolygon(wide_data, polygons):    
    
    wide_data["point_within_polygons"] = wide_data.POSITION_POINT.within(polygons)
    #Export to a second CSV file that only has the filtered results
    return wide_data.loc[wide_data['point_within_polygons'] == False]

# In[1]:
##########################################################################################################################################
debug_json_errors = False
chunksize = 1 if debug_json_errors else None
write_normalized_instead_of_wide = False

Mainpath = pl.Path(__file__).parent.resolve()    
source_data_directory_relative_path = "Source Data\\AirNav"
# extracted_jsons_directory = 'C:\\Users\\Lyle.Tripp\\Downloads'
extracted_jsons_directory = "C:\\Users\\Lyle.Tripp\\DOT OST\\volpe-org-324 - AEDT\\Lead emissions dispersion scoping\\Source Data\\AirNav\\Samples"
field_descriptions = pd.read_excel(pl.Path.joinpath(Mainpath,source_data_directory_relative_path,"AirNav_FlightHistory_DataDumpFormat_20211018.xlsx"), "Schedule",skiprows=3)
# field_descriptions["Row"] = field_descriptions.index
# field_descriptions.set_index("Row", inplace=True)
# field_descriptions.drop(axis=1,columns='Unnamed: 2',inplace=True)
position_object_index = field_descriptions[field_descriptions["Field Name"] == 'positions'].index.to_list().pop()
field_descriptions[field_descriptions['Description'].str.contains('takeoff')]
airnav_schedule_field_names_list = field_descriptions["Field Name"].drop(index = position_object_index, axis = 0).to_list()
airnav_positions_field_names_list = ['svd','lat','lon','fhd','fgs','fvr','so','sq','fal']

list_of_geo_fields_to_drop = ['POSITION_POINT', 'NEXT_POSITION_POINT', 'TAKEOFF_APT_REF_PT']

flightaware = {}
flightaware["input_filename_sans_ext"] = 'FlightAware_KCRQ_Sample_2022-04-04 (2)'
flightaware["input_file_extension"] = '.csv'
fa_input_filename = flightaware["input_filename_sans_ext"] + flightaware["input_file_extension"]
# fa = pd.read_csv(os.curdir+'\\Source Data'+'\\'+fa_input_filename)
# remap = MapOperationsColumns(fa, 'FlightAware')
# augmented_fa = AugmentPositionData(fa.rename(axis = 1, mapper = remap), 'FlightAware')
# augmented_fa_no_geom = augmented_fa.drop(list_of_geo_fields_to_drop, axis = 1)
# augmented_fa_no_geom.to_csv(os.path.join(os.curdir, 'Output/' + 'Modified_' + flightaware["input_filename_sans_ext"] + '.csv'), index = False)

#create one output file per input file
#ToDo: handle zip archives one at a time, delete temporary / extracted jsons immediately after use

json_filelist = []
for path, dirnames, filenames in os.walk(extracted_jsons_directory):
    for f in filenames:
        if f[-5:] == '.json':
            json_filelist.append(f)
bb_filter = {'lat1':32.5100, 'lon1':-118.1400, 'lat2':34.0800, 'lon2':-114.4300} #NAD83(NSRS2007) / California zone 6 (Google it) WGS84 Bounds: -118.1400, 32.5100, -114.4300, 34.0800
json_filelist =  [f for f in json_filelist] #
print(json_filelist)

# In[reading shapes]
polygons = gpa.read_file(pl.Path.joinpath(Mainpath, 'Source Data/KCRQ Airport/' + 'KCRQ_2 - Copy (2).kml'))

# In[1]
for filename in json_filelist:
    print(filename) #curious that a single file appears more than once in this list
    airnav = {}
    airnav["input_filename_sans_ext"] = filename[:-5]
    airnav["input_file_extension"] = '.json'
    airnav_input_filename = airnav["input_filename_sans_ext"] + airnav["input_file_extension"]
    airnav_filepath = os.path.abspath(extracted_jsons_directory+"\\"+airnav_input_filename)
    if debug_json_errors == False: 
        air = pd.read_json(airnav_filepath, lines = True, encoding_errors = 'ignore', nrows = 100)
        if write_normalized_instead_of_wide:
            PrepareAirNavData(air, airnav_input_filename, bb_filter = None, departure_airport = None, arrival_airport = 'KCRQ', write_normalized=True)
        else:                            
            wide_augmented_airnav = PrepareAirNavData(air, airnav_input_filename, bb_filter, None, None, False)
            wide_augmented_airnav.set_geometry('POSITION_POINT', crs = 3499, inplace = True)

            if polygons is not None:
                wide_augmented_airnav = filterWithPolygon(wide_augmented_airnav, polygons.loc[0,'geometry'])

            #ToDo: try using distance or length functions on projected coordinates https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.distance.html
                # e.g., wide_augmented_airnav = wide_augmented_airnav[[wide_augmented_airnav['Line to airport']]] 
            output_filepath = os.path.join(os.curdir, 'Output_arrivals/' + 'Modified_AirNav' + airnav["input_filename_sans_ext"] + '.csv')
            wide_augmented_airnav.to_csv(output_filepath, index = True)
            
            # del wide_augmented_airnav

        del air
    else:
        with pd.read_json(airnav_filepath, lines = True, encoding_errors = 'ignore', chunksize=chunksize) as reader:
            reader
            for chunk in reader:
                print(chunk)



##########################################################################################################################################

# # In[diagnostics]:
# fa_calls = pd.DataFrame(augmented_fa.loc[:,'cs'].reset_index(drop = True).to_frame().groupby('cs').indices.keys()).set_index(0, verify_integrity = True)
# airnav_calls = pd.DataFrame(augmented_airnav.loc[:,'cs'].reset_index(drop = True).to_frame().groupby('cs').indices.keys()).set_index(0, verify_integrity = True)
# print(fa_calls)
# print(airnav_calls)
# #is there anything in FA that's not in AirNav (we would do this in SQL with EXCEPT)
# print(len(fa_calls.join(airnav_calls, how='left')))
# print(len(fa_calls.join(airnav_calls, how='right')))
# print(len(fa_calls.join(airnav_calls, how='inner'))) 