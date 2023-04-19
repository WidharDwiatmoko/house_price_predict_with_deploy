
# California Houses Prices Prediction

## Brief Background

As we all know, California is one of the most famous state in the US. California has a lot of landmarks to visit, such as The famous Hollywood, Golden Gate Bridge, Silicon Valley, and many more. Therefore, there is always people from different state or abroad moving to California for Job, Education, Vacation, etc. With such high traffic from people coming to California, the demand of house also increase. especially people from abroad who knows very little of California, will have difficulty finding the best option for house. So I tried to create machine learning model to help end user to pick the best house based on few attributes

Link ML Deployment : https://house-price-predict.herokuapp.com/

## Dataset

[Calicornia House Price](https://www.kaggle.com/datasets/shibumohapatra/house-price)


## Dataset Reference

#### Predictors


| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `longitude` | `float` | **Required**. Longitude value for the block in California, USA  |
| `latitude` | `float` | **Required**. Latitude value for the block in California, USA  |
| `housing_median_age` | `int` | **Required**. Median age of the house in the block  |
| `total_rooms` | `int` | **Required**. Count of the total number of rooms (excluding bedrooms) in all houses in the block  |
| `total_bedrooms` | `int` | **Required**. Count of the total number of bedrooms in all houses in the block  |
| `population` | `int` | **Required**. Count of the total number of population in the block  |
| `households` | `int` | **Required**. Count of the total number of households in the block  |
| `median_income` | `float` | **Required**. Median of the total household income of all the houses in the block  |
| `ocean_proximity` | `categorical` | **Required**. Type of the landscape of the block `[ 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND' ]` |

#### target
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `median_house_value` | `float` | **Required**. Median of the household prices of all the houses in the block |

## Diagram Flow Preprocessing and EDA 

![Flow Prepro](https://github.com/WidharDwiatmoko/house_price_predict_with_deploy/blob/main/docs/Preprocess.png?raw=true)

## Diagram Flow Feature Engineering

![Flow FE](https://github.com/WidharDwiatmoko/house_price_predict_with_deploy/blob/main/docs/Feature Eng.png?raw=true)

## Diagram Flow Modeling and Deploying

![Flow model](https://github.com/WidharDwiatmoko/house_price_predict_with_deploy/blob/main/docs/modeling.png?raw=true)

## Running API Call

`POST API Call using Postman`

- URL

```url
https://house-price-predict.herokuapp.com
```
- Parameter

```url
/predict_api
```
- Payload

```json
  {
    "data":
    {
        "long": -122.23,	
        "lat": 37.88,
        "med_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "pop":	322.0,
        "hold":	126.0,
        "income": 8.3252,
        "ocean":"NEAR BAY"
    }
}
```

`Response API Call`
- success, **200 OK** 

![200 success](https://github.com/WidharDwiatmoko/house_price_predict_with_deploy/blob/main/docs/image_apiCall.png?raw=true)

`POST API Call using curl`

if you have no Postman installed on your machine, you can use `curl` to use `predict_api`. All you have to do just open your terminal and type in

```bash
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{
    "data":
    {
        "long": -122.23,
        "lat": 37.88,
        "med_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "pop":  322.0,
        "hold": 126.0,
        "income": 8.3252,
        "ocean":"NEAR BAY"
    }
}' \
 https://house-price-predict.herokuapp.com/predict_api
```

## Authors

- [@Widhar](https://github.com/WidharDwiatmoko)


