import xml.etree.ElementTree as ET
import csv

tree = ET.parse('./data/pollution.xml')
root = tree.getroot()

pollution_data = open('./data/pollution2.csv', 'w')
csv_writer = csv.writer(pollution_data)
air_data_head = []

count = 0
for member in root.findall('LuftDataObj'):
    air_data_instance = []
    address_list = []
    if count == 0:
        last_updated_col_name = member.find('LastUpdated').tag
        air_data_head.append(last_updated_col_name)
        pm10_col_name = member.find('PM10').tag
        air_data_head.append(pm10_col_name)
        count = count + 1

    last_updated = member.find('LastUpdated').text
    air_data_instance.append(last_updated)
    pm10 = member.find('PM10').text
    air_data_instance.append(pm10)
    csv_writer.writerow(air_data_instance)
pollution_data.close()
