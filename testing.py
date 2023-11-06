import requests

#host = 'potability-service-hthdd6e3ha-ew.a.run.app'  
#url = f'http://{host}:9090/predict'

url = 'https://potability-service-hthdd6e3ha-ew.a.run.app/predict'  # Replace with your actual service URL

water_source_id = 'H2O_Ivory_Coast_h402e'

water = {
    "ph": 9.05238368979066,
 "hardness": 120.02172614502156,
 "solids": 22203.487258877,
 "chloramines": 9.36833977172888,
 "sulfate": 285.2335275807391,
 "conductivity": 372.2752322143662,
 "organic_carbon": 14.538445861011496,
 "trihalomethanes": 64.1324260352138,
 "turbidity": 4.259719451958118
}


#requests.post(url, json=water)


response = requests.post(url, json=water).json()
print(response)


if response['potability'] == True:
    print(f'potability for water {water_source_id} is guaranteed: water can be commercialized and used for drinking usage!')
else:
    print(f'potability for water {water_source_id} is not guaranteed: water cannot be commercialized and drinking is NOT RECOMMENDED at the moment!')