<h2>AirBnb price prediction model</h2>

<p>This is a machine learning model whose purpose is to predict prices based on AirBnB listing features.  The purpose of this model is an experiment to determine how accurately one can determine a "reasonable" price of a short term rental given info such as number of rooms, amenities, and location.  NYC has been chosen in particular due to its large number and variety of listings and to reduce city-to-city or country-to-country data variability.  Full year data from 2019 was chosen to rule out the impact of the covid-19 pandemic, although a post-coronavirus study could be done with some modifications to the model or data pipeline.</p>

<p>During data exploration and cleaning, I created multiple models for average daily price predicted over certain time frames (year, winter, spring, summer, fall, and select months).  This experimentation is described in more detail <a href="https://docs.google.com/document/d/1gz1ZgBgyygkYrr9so776RsldtAkHHBHxNWZpARoW3rA/edit?usp=sharing">here</a>. Possible additional models that can be added in the future can include days around certain holidays or weekends vs. weekdays.</p>

<h3>Deploying the model</h3>

The model I decided to transition to a REST API and containerize is the predicted average daily price over a year.

To build the container image run the following in the same directory as the Dockerfile:

```
docker build -t <image name> .
```

Training is part of the image build process, so it will take a while depending on your machine specs.  The reason for training during the build is that Once the image is built it can be run under localhost and a chosen port with the following:

```
docker run -d -p <port>:9090 <image name>
```

Or the image can be deployed to a cloud provider such as AWS.

<h3>Model API</h3>

Once the container is running, you can test the model REST API can be reached at your chosen host (localhost or cloud provider).  You can send a test GET request through the following URLs:

```
<host>:<port>/api
```
returns a simple message about the model

```
<host>:<port>/api/test
```
returns a default prediction test case

To actually use the model, you need to send a POST request to:

```
<host>:<port>/api/predict
```

The body of the POST request should be JSON and needs to contain a subset of the following schema defined in input_data.py.  Default values are shown as well and will be used when they are not input by the user.  Defaults were calculated by mean or by highest value count.  Means were rounded to the nearest integer where they made sense (e.g. bedrooms vs. latitude):

    description: str = None
    host_is_superhost: int = 0
    latitude: float = 40.7282952399
    longitude: float = -73.9501208342
    accommodates: int = 3
    bathrooms: float = 1.0
    bedrooms: int = 1
    beds: int = 1
    entire_home_apt: int = 1
    all_day_check_in: int = 0
    toilet: int = 0
    accessible_height_bed: int = 0
    accessible_height_toilet: int = 0
    air_conditioning: int = 1
    air_purifier: int = 0
    baby_bath: int = 0
    baby_monitor: int = 0
    babysitter_recommendations: int = 0
    bathtub: int = 0
    bathtub_with_bath_chair: int = 0
    bbq_grill: int = 0
    beach_essentials: int = 0
    beachfront: int = 0
    bed_linens: int = 0
    breakfast: int = 0
    building_staff: int = 0
    buzzer_wireless_intercom: int = 0
    cable_tv: int = 0
    carbon_monoxide_detector: int = 1
    cats: int = 0
    ceiling_hoist: int = 0
    changing_table: int = 0
    childrens_books_and_toys: int = 0
    childrens_dinnerware: int = 0
    cleaning_before_checkout: int = 0
    coffee_maker: int = 0
    cooking_basics: int = 0
    crib: int = 0
    disabled_parking_spot: int = 0
    dishes_and_silverware: int = 0
    dishwasher: int = 0
    dogs: int = 0
    doorman: int = 0
    dryer: int = 0
    electric_profiling_bed: int = 0
    elevator: int = 0
    essentials: int = 1
    ethernet_connection: int = 0
    ev_charger: int = 0
    extra_pillows_and_blankets: int = 0
    extra_space_around_bed: int = 0
    family_kid_friendly: int = 0
    fire_extinguisher: int = 0
    fireplace_guards: int = 0
    firm_mattress: int = 0
    first_aid_kit: int = 0
    fixed_grab_bars_for_shower: int = 0
    fixed_grab_bars_for_toilet: int = 0
    flat_path_to_guest_entrance: int = 0
    free_parking_on_premises: int = 0
    free_street_parking: int = 0
    full_kitchen: int = 0
    game_console: int = 0
    garden_or_backyard: int = 0
    ground_floor_access: int = 0
    gym: int = 0
    hair_dryer: int = 1
    handheld_shower_head: int = 0
    hangers: int = 1
    heating: int = 1
    high_chair: int = 0
    host_greets_you: int = 0
    hot_tub: int = 0
    hot_water: int = 1
    hot_water_kettle: int = 0
    indoor_fireplace: int = 0
    internet: int = 0
    iron: int = 1
    keypad: int = 0
    kitchen: int = 1
    kitchenette: int = 0
    lake_access: int = 0
    laptop_friendly_workspace: int = 1
    lock_on_bedroom_door: int = 0
    lockbox: int = 0
    long_term_stays_allowed: int = 0
    luggage_dropoff_allowed: int = 0
    microwave: int = 0
    mobile_hoist: int = 0
    no_stairs_or_steps_to_enter: int = 0
    other: int = 0
    other_pets: int = 0
    outlet_covers: int = 0
    oven: int = 0
    pack_n_play_travel_crib: int = 0
    paid_parking_off_premises: int = 0
    paid_parking_on_premises: int = 0
    patio_or_balcony: int = 0
    pets_allowed: int = 0
    pets_live_on_this_property: int = 0
    pocket_wifi: int = 0
    pool: int = 0
    pool_with_pool_hoist: int = 0
    private_bathroom: int = 0
    private_entrance: int = 0
    private_living_room: int = 0
    refrigerator: int = 0
    room_darkening_shades: int = 0
    safety_card: int = 0
    self_check_in: int = 0
    shampoo: int = 1
    shower_chair: int = 0
    single_level_home: int = 0
    ski_in_ski_out: int = 0
    smart_lock: int = 0
    smoke_detector: int = 1
    smoking_allowed: int = 0
    stair_gates: int = 0
    step_free_shower: int = 0
    stove: int = 0
    suitable_for_events: int = 0
    table_corner_guards: int = 0
    tv: int = 1
    washer: int = 0
    washer_dryer: int = 0
    waterfront: int = 0
    well_lit_path_to_entrance: int = 0
    wheelchair_accessible: int = 0
    wide_clearance_to_shower: int = 0
    wide_doorway_to_guest_bathroom: int = 0
    wide_entrance: int = 0
    wide_entrance_for_guests: int = 0
    wide_entryway: int = 0
    wide_hallways: int = 0
    wifi: int = 1
    window_guards: int = 0
    pred_price_year_avg_bin: int = 2
    available_year_avg: float = 0.309226983
    min_nights_year_avg: float = 5.5922012336

Example of request body and API response (tested through Postman, but a simple curl command would work as well):

Request:
```json
{
	"bedrooms": 1,
    "latitude": 40.7382952399,
    "longitude": -73.9601208342,
	"bathrooms": 2.5,
	"all_day_check_in": 1,
	"accommodates": 4
}
```

Response:
```json
{
    "pred_price_year_avg": "218.5152"
}
```

<h3>Future improvements</h3>

Currently retraining can be done by replacing the dataset url in airbnb.py.  This can be streamlined to a process that is run regularly (perhaps weekly or monthly) as new data is available.  Unfortunately, the <a href="http://insideairbnb.com/get-the-data.html">data source</a> does not appear to have an API to get the latest data.  Maybe a date-based process can be implemented to check for updated data sets.

Suggestions for other fixes and improvements are welcome as well!