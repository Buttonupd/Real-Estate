from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from predict.models import PredResults
import numpy as np


def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):
    if request.POST.get('action') == 'post':
        # Receive data from client

        No = request.POST.get('No')
        X1_transaction_date = request.POST.get('X1_transaction_date')
        X2_house_age = request.POST.get('X2_house_age ')
        X3_distance_to_nearest_MRT_station = request.POST.get('X3_distance_to_nearest_MRT_station')
        X4_number_of_convenience_stores = request.POST.get('X4_number_of_convenience_stores')
        X5_latitude = request.POST.get('X5_latitude')
        X6_longitude = request.POST.get('X6_longitude')
        Y_house_price_of_unit_area = request.POST.get('Y_house_price_of_unit_area')

        # Unpickle model
        model = pd.read_pickle("ENB.pickle")

        # Make prediction
        result = float or int(model.predict([[Y_house_price_of_unit_area]]))
        # np.any.isnan(result)
        # np.all(np.isfinite(result))

        predictions = result

        PredResults.objects.create(No=No, Y_house_price_of_unit_area=Y_house_price_of_unit_area,
                                   X1_transaction_date=X1_transaction_date, X2_house_age=X2_house_age,
                                   X3_distance_to_nearest_MRT_station=X3_distance_to_nearest_MRT_station,
                                   X4_number_of_convenience_stores=X4_number_of_convenience_stores,
                                   X5_latitude=X5_latitude, X6_longitude=X6_longitude, predictions=predictions

                                   )

        return JsonResponse({'result': predictions, 'No': No,
                             'X1_transaction_date': X1_transaction_date, 'X2_house_age': X2_house_age,
                             'X3_distance_ato_nearest_MRT_station': X3_distance_to_nearest_MRT_station,
                             'X4_number_of_convenience_stores': X4_number_of_convenience_stores,
                             'X5_latitude': X5_latitude, 'X6_longitude': X6_longitude,
                             'Y_house_price_of_unit_area': Y_house_price_of_unit_area},
                            )


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}

    return render(request, "results.html", data)
