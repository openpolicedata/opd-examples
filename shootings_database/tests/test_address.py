import pytest

# TODO: Streets that are directions or single letter
# '1100 Park Anue\nOrange Park, Florida'
# President George Bush Hwy
# 'SE 82nd Ave & SE Monterey (Clackamas County)'
# St Helens Way
# '1700 WEST AVE K'
# '1700 WEST AVE K-4'
# 'I-5, 1 Mile North of Templin Hwy'
# '710 Fwy transition from W / B 105 Fwy, Lynwood'
# 'S / E CORNER OF IMPERIAL HWY  /  COMPTON AVE'
# 'SPUNKY CANYON ROAD, AT SOUTHERN CALIFORNIA EDISON (SCE) #898688E'
# 'East 10th Street, between Lasalle and Olney streets'
# 'OLD CONCORD ROAD, CHARLOTTE 28213'
# 'W 35th St/W 34th St (3500 CRAWFORD AVE)'
# '6600 S MoPac Expy Svrd SB'
@pytest.mark.parametrize('number', ['1','1234','34XX'])
def test_address():
    pass