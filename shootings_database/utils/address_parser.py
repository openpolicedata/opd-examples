from typing import Optional
import usaddress
import json
import pandas as pd
import re
from collections import OrderedDict
from openpolicedata.defs import states as states_dict
from openpolicedata.utils import split_words

STREET_NAMES = usaddress.STREET_NAMES
STREET_NAMES.update(['la','bl','exwy','anue','corridor', 'service rd'])
# Not sure that these are ever street names
STREET_NAMES.remove('fort') 
STREET_NAMES.remove('center')
STREET_NAMES.remove('lake')
STREET_NAMES.remove('camp')

_states = list(states_dict.keys())
for v in states_dict.values():
    _states.append(v)

def find_address_col(df_test: pd.DataFrame, error: str='ignore'):
    """Find address or street column if it exists

    Parameters
    ----------
    df_test : pd.DataFrame
        Table to find column in 
    error : str, optional
        If 'raise', an exception will be thrown if an unexpected exception is thrown when trying to ID a column
        If 'ignore', no exception will be thrown, the column that was running when the exception occurred will not be ID'ed as an address column 
        , by default 'ignore'

    Returns
    -------
    list[str]
        List of address or street columns
    """
    addr_col = [x for x in df_test.columns if "LOCATION" in x.upper()]
    for col in addr_col:
        try:
            # Check if location is location on the body
            if df_test[col].apply(lambda x: 
                                  False if not isinstance(x,str) else
                                   any([y in x.lower() for y in ['arm','torso','back','chest','hand','leg','abdomen','neck','throat','head']])
                                   ).mean() > 0.8:
                continue
            tags = df_test[col].apply(lambda x: tag(x, col, error='ignore'))
        except AttributeError as e:
            if (error=='ignore') or (len(e.args)>0 and e.args[0]=="'bool' object has no attribute 'strip'"):
                continue
            else:
                raise
        except ValueError as e:
            if (error=='ignore') or (str(e)=='Data appears to be an (x,y) location and not an address'):
                continue
            else:
                raise
        except:
            if error=='ignore':
                continue
            else:
                raise
        tags = tags[tags.apply(lambda x: isinstance(x[1],str) and x[1]!='Null')]
        if tags.apply(lambda x: x[1]).isin(['Street Address','Intersection','Block Address', 'Street Name', 
                                            'StreetDirectional', 'County', 'Building', 'Bridge']).all():
            return [col]

    addr_col = [x for x in df_test.columns if x.upper() in ["STREET"] or x.upper().replace(" ","").startswith("STREETNAME")]
    if len(addr_col):
        return addr_col
    addr_col = [x for x in df_test.columns if 'address' in split_words(x,case='lower')]
    return addr_col

_default_delims = ['^', r"\s", "$", ',']

# Based on https://stackoverflow.com/questions/30045106/python-how-to-extend-str-and-overload-its-constructor
class ReText(str):
    @staticmethod
    def __construct_string(value, name, opt, lookahead):
        if isinstance(value, list):
            value_list = value
            value = ''
            for v in value_list:  # Append values in list
                if isinstance(v,str):
                    value+=v
                else:
                    # List of possible values
                    value+=rf'({"|".join(sorted(list(v), key=len, reverse=True))})'

        value = str(value)        
        if name != None:
            value = rf"(?P<{name}>{value}"
            value+=r"(?=(" + "|".join(lookahead) +")))"

            if opt and value[-1]!='?':
                value+=r"?" 
        elif opt and value[-1]!='?':
            if value[0]!='(':
                value = '('+value+')'
            else:
                # Check if string is enclose in parentheses
                num_open = 0
                slash_last = False
                for c in value[:-1]:
                    if c=='(' and not slash_last: # Ignore r'\('
                        num_open+=1
                    elif c==')' and not slash_last:  # Ignore r'\)'
                        num_open-=1
                        if num_open==0:
                            # Parenthesis at beginning does not enclose entire string
                            value = '('+value+')'
                    slash_last = c=='\\'

                    
            value+=r"?"

        return value

    def __new__(cls, value, name=None, opt=False, delims=_default_delims, lookahead=None):
        lookahead = lookahead if lookahead else delims

        orig_value = value

        value = ReText.__construct_string(value, name, opt, lookahead)
        
        # explicitly only pass value to the str constructor
        self = super(ReText, cls).__new__(cls, value)
        self.name = name
        self.opt = opt
        delims = delims.copy()
        lookahead = lookahead.copy()
        self.delims = delims
        self.lookahead = lookahead
        self.orig_value = orig_value
        return self
    

    def add_delim(self, delims, add_to_lookahead=True):
        if isinstance(delims, str):
            delims = [delims]

        new_delims = self.delims.copy()
        new_delims.extend(delims)
        new_look = self.lookahead.copy()
        if add_to_lookahead:
            new_look.extend(delims)

        return ReText(self.orig_value, self.name, self.opt, new_delims, new_look)
    
    
    def change_opt(self, opt):
        return ReText(self.orig_value, self.name, opt, self.delims, self.lookahead)


    def __add__(self, other):
        x = str(self)
        if isinstance(other, ReText):
            # This can probably be handled better in the case where ReText are nested
            if len(self.delims)>0:
                x+=r"("+ "|".join(self.delims) + ")"
                if other.opt:
                    x+=r"*"
                else:
                    x+=r"+"
            return ReText(x+str(other), opt=other.opt, delims=other.delims)
        else:
            return x+other
        
    def __radd__(self, other):
        return  ReText(other+str(self), opt=self.opt, delims=self.delims)
        
    def ordinal(self, ord):
        ord = 'Second' if ord==2 else ord
        return ReText(self.replace(r"(?P<", r"(?P<"+ord), opt=self.opt, delims=self.delims)

_opt_place4 = ReText(r"[a-z ]+", 'PlaceName')
_building_name = ReText([r'[a-z]+\s+[a-z ]+\s+',['center','hospital','shelter','motel','resort']], 'BuildingName')
_opt_near = ReText('near', 'Distance', opt=True)
_opt_building_name = ReText(_building_name, opt=True)
_p_building = re.compile("^"+_building_name+"$", re.IGNORECASE)
_p_building2 = re.compile("^"+_building_name+_opt_near.change_opt(True)+_opt_place4+"$", re.IGNORECASE)

_bridge_name = ReText(r'[a-z ]+\sbridge', 'BridgeName')
_over = ReText('over', 'Preposition')
_body_of_water = ReText([r'[a-z ]+\s',['creek','river']], 'BodyOfWater')
_p_bridge = re.compile("^"+_bridge_name+_over+_body_of_water+"$", re.IGNORECASE)

# Prevent street names from being in place name    
_opt_place = ReText(r"(?<=\s)((?!(?<=\s)("+"|".join(STREET_NAMES)+r")\s)[a-z ])+", 'PlaceName', delims=[r'\s',','])
_opt_state = ReText([_states], 'StateName')
_opt_zip = ReText(r'\d{5}+', 'ZipCode', opt=True)
_opt_place_line = ReText("("+_opt_place+_opt_state+")",opt=True)+_opt_zip

_p_zip_only = re.compile("^"+_opt_zip.change_opt(False)+"$")

_post_type_delims = _default_delims.copy()
_post_type_delims.extend([r'\n'])
_block_num = ReText(r"\d+[0X]{2}", "BlockNumber")
_block_ind = ReText([["Block of",'BLK', 'block', 'of']], "BlockIndicator")
_block_ind2 = ReText('between', "BlockIndicator")
_opt_street_dir = ReText([usaddress.DIRECTIONS, r'\.?'], 'StreetNamePreDirectional', opt=True)
_street_name = ReText(r"i?[\w \.\-']+?", "StreetName")  # i- for interstates

_pre_street_names = STREET_NAMES.copy()
_pre_street_names.update(['ih','route','sr','fm','interstate','spur'])
_pre_street_names.remove("st")
_pre_street_names.remove("la")
_pre_street_names.remove("garden")

_directions_expanded = list(usaddress.DIRECTIONS.copy())
_directions_expanded.extend(['northbound','southbound','eastbound','westbound',r'n\s*/?\s*b',r's\s*/?\s*b',r'w\s*/?\s*b',r'e\s*/?\s*b'])

_hwy_types = _states.copy()
_hwy_types.extend(['indian','state','county'])
_pre_type = r"("+ReText([_hwy_types])+r'(?=\s))?\s*'+str(ReText([_pre_street_names, r'\.?']))
_opt_pre_type = r'\s*('+ReText(_pre_type, 'StreetNamePreType')+r"(?!\s("+"|".join(STREET_NAMES)+r")))?\s*"
_opt_post_type = ReText([STREET_NAMES, r'\.?'], 'StreetNamePostType', opt=True, delims=_post_type_delims)
_opt_post_dir = ReText([_directions_expanded, r'\.?'], 'StreetNamePostDirectional', opt=True, delims=_post_type_delims)
_opt_service_road = ReText([['svrd','service road']], 'StreetNameServiceRoadIndicator', opt=True)
_street_match = _opt_street_dir+_opt_pre_type+_street_name+_opt_post_type+_opt_service_road+_opt_post_dir
_opt_address_num = ReText(r"[\dX]+", 'AddressNumber', opt=True)
_street_match_w_addr = _opt_address_num+_street_match

_hwy_types.append('us')
_numeric_street_name = ReText(r'\d+[nsew]?', 'StreetName')
_us_hwy = _opt_address_num+ReText([_hwy_types, r'\s*', '(hwy)?'], 'StreetNamePreType')+_numeric_street_name

_and_str_block = ReText([['and',r'&']], 'BlockRangeSeparator')
_opt_cross_street = ReText((_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir).replace(r"(?P<StreetName", r"(?P<CrossStreetName"), opt=True)
_p_block = re.compile("^"+_block_num+_block_ind+_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir+
                      _opt_place_line+"$", re.IGNORECASE)
_p_block2 = re.compile("^"+_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir+_block_ind2+
                       (_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir).replace(r"(?P<StreetName", r"(?P<CrossStreetName")+_and_str_block+
                       (_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir).replace(r"(?P<StreetName", r"(?P<SecondCrossStreetName")+
                       "$", re.IGNORECASE)
_p_block3 = re.compile("^"+_block_num+_block_ind+_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir+
                      _opt_place_line+_opt_near+_opt_cross_street+"$", re.IGNORECASE)

_slash = ReText(r"\s*/\s*", 'BlockRangeSeparator',delims=[r'\s',r'\w'])
_cross_street = ReText((_opt_street_dir+_street_name.add_delim("/")+_opt_post_type.add_delim("/")+_opt_post_dir.add_delim("/")).replace(r"(?P<StreetName", r"(?P<CrossStreetName"),
                       delims=[r'\s','/'])
_p_block4 = re.compile("^"+_cross_street+_slash+
                       (_opt_street_dir+_street_name+_opt_post_type+_opt_post_dir).replace(r"(?P<StreetName", r"(?P<SecondCrossStreetName")+
                       r'\s+\('+_block_num+_opt_street_dir+_street_name.add_delim(r'\)')+_opt_post_type.add_delim(r'\)')+_opt_post_dir.add_delim(r'\)')+r'\)'
                       "$", re.IGNORECASE)

_dir2 = ReText([_directions_expanded], 'Direction')
_p_block_service_road = re.compile("^"+_block_num+_opt_street_dir+_street_name+_opt_post_type+_opt_service_road.change_opt(False)+
                                   _dir2.change_opt(False)+"$", re.IGNORECASE)

_dir = ReText([usaddress.DIRECTIONS, r'\.?'], 'Direction')
_opt_dist = ReText(r"\d+ miles?", 'Distance', opt=True)
_of = ReText('of', 'Preposition')
_p_directional = re.compile('^'+_street_match+_opt_dist+_dir+_of+_street_match.replace(r"(?P<StreetName", r"(?P<CrossStreetName")+"$", re.IGNORECASE)

_and_str = ReText([['and',r'&', r'/', 'at', r'near( the)?']], 'IntersectionSeparator')
_county_delims = _default_delims.copy()
_county_delims.append(r'\)')
  # Place name in parentheses
_opt_place_in_paren = ReText(r"(\("+ReText(r"[a-z \.\n\-]+",'PlaceName', delims=_county_delims)+r"\))", opt=True, delims=["$"])
_p_intersection_hwy = re.compile("^"+_us_hwy+_and_str+
                             (_street_match_w_addr).ordinal(2)+
                             _opt_place_line+"$", re.IGNORECASE)
_us_hwy = _opt_address_num+ReText([_hwy_types, r'\s*', '(hwy)?'], 'StreetNamePreType')+_numeric_street_name
_p_intersection_hwy2 = re.compile("^"+_opt_pre_type+_numeric_street_name+_and_str+
                             (_opt_pre_type+_numeric_street_name).ordinal(2)+
                             _opt_place_line+"$", re.IGNORECASE)
_p_intersection = re.compile("^"+_street_match_w_addr+_and_str+
                             (_street_match_w_addr).ordinal(2)+
                             _opt_place_line+"$", re.IGNORECASE)
_p_intersection2 = re.compile("^"+_street_match_w_addr+_and_str+
                             (_street_match_w_addr).ordinal(2)+
                             _opt_place_in_paren+"$", re.IGNORECASE)
_opt_place_end = ReText(r"(?<=\s)((?!(?<=\s)("+"|".join(STREET_NAMES)+r")\s)[a-z ])+", 'PlaceName', delims=['$'])
_p_intersection4 = re.compile("^"+_street_match_w_addr+_and_str+
                             (_street_match_w_addr).ordinal(2)+r",\s*"+
                             _opt_place_end+"$", re.IGNORECASE)
_int_type = ReText("transition from", "IntersectionType")
_opt_place2 = ReText(r"(?<=\s)[a-z ]+", 'PlaceName')
_p_intersection3 = re.compile("^"+_street_match+_int_type+_dir2+_street_match.ordinal(2)+_opt_place2+"$", re.IGNORECASE)

_corners = ['northeast','northwest','southeast','southwest',r'n\s*/?\s*e',r'n\s*/?\s*w',r's\s*/?\s*e',r's\s*/?\s*w']
_dir3 = ReText([_corners], 'Direction')
_int_type2 = ReText("corner", "IntersectionType")
_p_intersection5 = re.compile("^"+_dir3+_int_type2+_of+_street_match_w_addr+_and_str+
                             (_street_match_w_addr).ordinal(2)+"$", re.IGNORECASE)
_p_intersection6 = re.compile("^"+_dir+_of+_street_match_w_addr+ReText('on the','IntersectionSeparator')+
                              _opt_service_road.ordinal(2).change_opt(True)+_of.ordinal(2)+
                             ReText(_opt_pre_type).ordinal(2)+_numeric_street_name.ordinal(2)+"$", re.IGNORECASE)

occ_delims = _default_delims.copy()
occ_delims.append(",")
occ_look = occ_delims.copy()
occ_look.extend([r"#",r"\d"])
_opt_occupancy_type = ReText([['APT','apartment']], 'OccupancyType',opt=True,lookahead=occ_look)
_opt_occupancy_id = ReText(r"#?\s?(?<=(APT\s|.APT|..[\sT]#|T\s#\s))[a-z]?\-?\d+", 'OccupancyIdentifier', opt=True)
_p_hwy_address = re.compile("^"+_us_hwy+_opt_place_line+"$",re.IGNORECASE)
_p_address = re.compile("^"+_opt_building_name+_opt_address_num+_opt_building_name.ordinal(2)+_street_match+
                        _opt_occupancy_type+_opt_occupancy_id+
                        _opt_place_line+"$", re.IGNORECASE)
_p_address2 = re.compile("^"+_opt_building_name+_street_match_w_addr+
                        _opt_occupancy_type+_opt_occupancy_id+
                        _opt_place_in_paren+"$", re.IGNORECASE)
_opt_place3 = ReText(r"(?<=\s)[a-z]+", 'PlaceName')
_p_address3 = re.compile("^"+_opt_building_name+_opt_address_num+_opt_building_name.ordinal(2)+_street_match+
                        _opt_occupancy_type+_opt_occupancy_id+r",\s*"+
                        _opt_place3+_opt_zip+"$", re.IGNORECASE)

_address_ambiguous = ReText(r"space [\da-z]+", 'Ambiguous')
_p_address_w_ambiguous = re.compile("^"+_opt_address_num+_street_match+_address_ambiguous+"$",re.IGNORECASE)

_address_ambiguous2 = ReText(".+", 'Ambiguous')
_p_street_plus_ambiguous = re.compile('^'+_street_match+", at"+_address_ambiguous2+"$", re.IGNORECASE)

_multiple_address = re.compile(r"Location #\d+\s"+_street_match_w_addr, re.IGNORECASE)

_p_county = re.compile("^"+ReText(r"[a-z]+\s*[a-z]*\sCounty", 'PlaceName')+"$", re.IGNORECASE)
_p_unknown = re.compile("^"+ReText(r"unknown\b.+",'Unknown')+"$", re.IGNORECASE)

_latitude = ReText([r"\-?", [str(x) for x in range(0,91)], r'\.\d+'], 'Latitude')
_longitude = ReText([r"\-?", [str(x) for x in range(0,181)], r'\.\d+'], 'Longitude')
_p_coords = re.compile("^"+_latitude+_longitude+"$", re.IGNORECASE)
_p_plus_code = re.compile("^"+ReText(r"^[a-z0-9]+\+[a-z0-9]{2}", 'PlusCode')+"$", re.IGNORECASE)


def _clean_groupdict(m, error='ignore'):
    m = m.groupdict()
    s = dir = None
    for k,v in m.items():
        if 'Directional' in k:
            dir = (k,v)
            s = None
        elif "PostType" in k:
            if  v==None and (dir!=None and dir[1]!=None) and (s!=None and s[1].lower() in STREET_NAMES):
                # Street name is a direction which fooled the regular expression
                m[k] = s[1]
                m[s[0]] = dir[1]
                m[dir[0]] = None
            s = dir = None
        elif "StreetName" in k:
            s = (k,v)
        else:
            s = dir = None

        if v and k=='SecondBuildingName':
            if error=='raise' and 'BuildingName' in m and m['BuildingName']:
                raise NotImplementedError()
            m['BuildingName'] = v
            m[k] = None

    return OrderedDict({k:v for k,v in m.items() if v is not None})


def _address_search(p, x, error='ignore'):
    if isinstance(p, list):
        for y in p:
            if (m:=_address_search(y, x, error=error)):
                return m
        return m
    if m:=p.search(x):
        m = _clean_groupdict(m, error=error)

    return m

def _check_result(result, usa_result, col_name=None):
    if result==usa_result:
        return True
    if result[1]=='Intersection':
        if usa_result[1]=='Ambiguous' and 'IntersectionSeparator' in result[0] or \
            'near' in result[0]['IntersectionSeparator'].lower():
            # usaddress misintepretation of street intersection without post type
            # such as 'Columbia and North Fessenden'
            return True
        if 'StreetName' not in usa_result[0] or usa_result[0]['StreetName'].lower().startswith('and') or \
            ' and ' in usa_result[0]['StreetName'].lower() or \
            result[0]['IntersectionSeparator']=='at' or \
            ('StreetNamePreType' in result[0] and result[0]['StreetNamePreType'].title() in _states) or \
            (result[0]['IntersectionSeparator']=='/' and ('IntersectionSeparator' not in usa_result[0] or usa_result[0]['IntersectionSeparator']!='/')) or \
            (usa_result[1]=='Intersection' and 'SecondStreetName' not in usa_result[0] and 'Recipient' in usa_result[0]):
            # usaddress missed street OR
            # usaddress included separator in street name OR
            # usaddress does not recognize / as separator
            return True
        # Check if usaddress included direction in street name
        dir = None
        for k,v in result[0].items():
            if k in usa_result[0] and usa_result[0][k] in [v, "("+v+")"]:
                pass
            elif "Directional" in k and k not in usa_result[0]:
                dir = v
                continue
            elif "StreetName" in k and dir!=None and usa_result[0][k]==dir+" "+v:  # Check if direction included with street
                pass
            else:
                return False
            dir = None

        return True
    elif result[1]=='Street Address':
        if 'StreetNameServiceRoadIndicator' in result[0]:
            return True
        if (usa_result[1]=='Ambiguous' and 'Recipient' in usa_result[0] and usa_result[0]['Recipient'].endswith('Hwy')) or \
            (usa_result[1]=='Ambiguous' and list(usa_result[0].keys()) == ['Recipient'] and \
                col_name and col_name.lower() in ['street','street name'] and list(result[0].keys()) == ['StreetName']) or \
            any([x in usa_result[0] and '\n' in usa_result[0][x] for x in ["StreetName",'PlaceName','StreetNamePostType','OccupancyIdentifier']]) or \
            (all([x in usa_result[0].values() for x in result[0].values()]) and list(usa_result[0].keys())[-1]=='OccupancyIdentifier') or \
            (list(usa_result[0].keys())==['BuildingName'] and all([x in result[0].keys() for x in ['StreetName','StreetNamePostType']])) or \
            ('StateName' in usa_result[0] and usa_result[0]['StateName'] in ['INN', 'INN)']) or \
            ("Ambiguous" in result[0] and result[0]['Ambiguous'].lower().startswith('space')) or \
            ('StreetNamePreType' in result[0] and result[0]['StreetNamePreType'].lower().startswith('indian')) or \
            ('StreetNamePreType' in usa_result[0] and usa_result[0]['StreetNamePreType'].lower()=='camino') or \
            ('StreetNamePostType' in result[0] and result[0]['StreetNamePostType']=='Service Rd') or \
            ('AddressNumber' in usa_result[0] and not usa_result[0]['AddressNumber'].isdigit() and not usa_result[0]['AddressNumber'].lower().endswith('x')):
            # usa_address unable to get President George Bush Hwy
            # usa_address has trouble with \n
            return True
        skip_state = False
        skip_next = False
        for k,v in result[0].items():
            if skip_next:
                skip_next = False
                continue
            # usaddress might include newline character as part of name
            k_usa = "Recipient" if k=='BuildingName' else k
            k_usa = 'Recipient' if k_usa=='PlaceName' and 'PlaceName' not in usa_result[0] else k_usa
            if k_usa not in usa_result[0] or usa_result[0][k_usa] not in [v, v+'\n', "("+v.replace('- ','')+")"]:
                if k=="PlaceName" and k in usa_result[0] and "StateName" in result[0] and \
                    "StateName" not in usa_result[0] and usa_result[0][k]==v+" "+result[0]['StateName']:
                    # usaddress included state with place name
                    skip_state = True
                elif (skip_state and k=="StateName") or \
                    (k=='OccupancyIdentifier' and k in usa_result[0] and v.replace("#","# ")==usa_result[0][k]):
                    pass
                elif k.endswith("StreetName") and k in usa_result[0] and (m:=k+"PreDirectional") in usa_result[0] and \
                    m not in result[0] and usa_result[0][m].lower() not in usaddress.DIRECTIONS and \
                    v==usa_result[0][m]+" "+usa_result[0][k]:  # Non-directional value marked as directional by usaddress
                    pass
                elif k.endswith("StreetName") and k in usa_result[0] and usa_result[0][k]!= v and v.startswith(usa_result[0][k]) and \
                    (m:=k+"PostType") in result[0] and m in usa_result[0] and usa_result[0][m].lower() not in STREET_NAMES and \
                    (v+" "+result[0][m]).replace(usa_result[0][k]+" ","") == usa_result[0][m]:
                    # usaddress put part of street name in post type
                    skip_next = True
                elif k=='PlaceName' and 'PlaceName' not in usa_result[0] and "StateName" not in result[0] and "StateName" in usa_result[0] and \
                    usa_result[0]['StateName']==v and v.title() not in _states:
                    # usaddress mistakenly called this a state
                    pass
                elif k=='StreetName' and "StreetNamePreModifier" not in result[0] and "StreetNamePreModifier" in usa_result[0] and \
                    v==usa_result[0]['StreetNamePreModifier']+" "+usa_result[0][k]:
                    pass
                else:
                    return False
            
        return True
    else:
        raise NotImplementedError()


def tag(address_string: str, col_name: Optional[str]=None, error:str='ignore'):
    """Split address into labeled components (address number, street name, etc.)

    Parameters
    ----------
    address_string : str
        String containing address
    col_name : Optional[str], optional
        Optional name of column that contained address_string. Only used if error='raise', by default None
    error : str, optional
        'raise' or 'ignore'. Whether to throw an error if either:
        1. tagged result does not equal result produced by the usaddress package OR
        2. tagged result is not recognized as a known correction to a usaddress package result
        , by default 'ignore'

    Returns
    -------
    dict
        Dictionary containing tags and values for parsed address_string
    str
        Type assigned to street address string (Address, Block Address, Intersection, etc.)
    """
    
    assert error in ['raise','ignore']
    if pd.isnull(address_string):
        return ({}, "Null")
    elif isinstance(address_string, dict):
        if 'human_address' not in address_string.keys():
            if sorted(address_string.keys())==['x','y']:
                raise ValueError("Data appears to be an (x,y) location and not an address")
            raise KeyError("Unknown dictionary keys for address")
        address_dict = json.loads(address_string['human_address'])
        result = tag(address_dict['address'], col_name, error)
        if error=='raise' and not all([x in ['address','city','state','zip'] for x in address_dict]):
            raise NotImplementedError()
        for k, kout in zip(['city','state','zip'], ['PlaceName','StateName','ZipCode']):
            if k in address_dict and len(address_dict[k]):
                result[0][kout] = address_dict[k]
        return result
    
    # Ensure that /'s are surrounded by spaces
    address_string = address_string.strip().replace("/"," / ")
    m1=m2=m3=m4=m5=None
    if m1:=_address_search(_p_unknown, address_string, error=error):
        return (m1, 'Null')
    if m1:=_address_search(_p_zip_only, address_string, error=error):
        return _get_address_result(m1, "Ambiguous", error=error)
    if (m1:=_address_search(_p_block, address_string, error=error)) or \
        (m2:=_address_search(_p_block2, address_string, error=error)) or \
        (m3:=_address_search(_p_block3, address_string, error=error)) or \
        (m4:=_address_search(_p_block4, address_string, error=error)) or \
        (m5:=_address_search(_p_block_service_road, address_string, error=error)):
        return _get_address_result([m1,m2,m3,m4,m5], "Block Address", error=error)
    elif (m1:=_address_search(_p_directional, address_string, error=error)):
        return (m1, "StreetDirectional")
    elif (m:=_address_search(_p_building, address_string, error=error)) or \
        (m:=_address_search(_p_building2, address_string, error=error)):
        return _get_address_result(m, "Building", address_string, type_check='Ambiguous', error=error)
    elif m:=_address_search([_p_intersection_hwy,_p_intersection_hwy2, _p_intersection, _p_intersection2,
                             _p_intersection3, _p_intersection4, _p_intersection5, _p_intersection6], 
                            address_string):
        return _get_address_result(m, "Intersection", address_string, error=error)
    elif m1:=_address_search(_p_county, address_string, error=error):
        return (m1, "County")
    elif (m1:=_address_search(_p_bridge, address_string, error=error)):
        return _get_address_result(m1, "Bridge", address_string, type_check='Ambiguous', error=error)
    elif (m:=_address_search(_p_hwy_address, address_string, error=error)) or \
        (m:=_address_search(_p_address, address_string, error=error)) or \
        (m:=_address_search(_p_address2, address_string, error=error)) or \
        (m:=_address_search(_p_address_w_ambiguous, address_string, error=error)) or \
        (m:=_address_search(_p_address3, address_string, error=error)):
        return _get_address_result(m, "Street Address", address_string, col_name=col_name, error=error)
    elif m1:=[_clean_groupdict(x) for x in re.finditer(_multiple_address, address_string)]:
        results = [_get_address_result(x, "Street Address", error=error) for x in m1]
        return [x[0] for x in results], [x[1] for x in results]
    elif m1:=_address_search(_p_street_plus_ambiguous, address_string, error=error):
        return _get_address_result(m1, "Street Address", address_string, check_ambiguous=True, error=error)
    elif m:=_address_search(_p_coords, address_string, error=error):
        return _get_address_result(m, "Coordinates", error=error)
    elif m:=_address_search(_p_plus_code, address_string, error=error):
        return _get_address_result(m, "PlusCode", error=error)
    elif error=='raise':
        raise NotImplementedError()
    else:
        return usaddress.tag(address_string)
    
def _get_address_result(results, name, address_string=None, type_check=None, col_name=None, check_ambiguous=False, error='ignore'):
    if not isinstance(results, list):
        results = [results]
    for r in results:
        if r:
            result = [r, name]
            break
    else:
        raise ValueError("No result found")

    if address_string or type_check:     
        try:
            usa_result = usaddress.tag(address_string)
        except usaddress.RepeatedLabelError as e:
            return result
        except:
            raise
        if type_check:
            if error=='raise' and usa_result[1]!=type_check:
                raise NotImplementedError()
        elif check_ambiguous and usa_result[1]=='Ambiguous':
            pass
        elif error=='raise' and address_string and not _check_result(result, usa_result, col_name):
            raise NotImplementedError()
    return result