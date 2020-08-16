from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults


def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):
    if request.POST.get('action') == 'post':
        # Receive data from client

        No = float(request.POST.get('No'))
        X1_transaction_date = float(request.POST.get('X1_transaction_date'))
        X2_house_age = float(request.POST.get('X2_house_age'))
        X3_distance_to_nearest_MRT_station = float(request.POST.get('X3_distance_to_nearest_MRT_station'))
        X4_number_of_convenience_stores = float(request.POST.get('X4_number_of_convenience_stores'))
        X5_latitude = float(request.POST.get('X5_latitude'))
        X6_longitude = float(request.POST.get('X6_longitude'))
        Y_house_price_of_unit_area = float(request.POST.get('Y_house_price_of_unit_area'))

        # Unpickle model
        model = pd.read_pickle("Valuation.pickle")
        # Make prediction
        result = model.predict([[Y_house_price_of_unit_area, No, X1_transaction_date,
                                 X2_house_age, X3_distance_to_nearest_MRT_station, X4_number_of_convenience_stores,
                                 X5_latitude, X6_longitude




                                 ]])

        classification = result[0]

        PredResults.objects.create(No=No, X1_transaction_date=X1_transaction_date, X2_house_age=X2_house_age,
                                   X3_distance_to_nearest_MRT_station=X3_distance_to_nearest_MRT_station,
                                   X4_number_of_convenience_stores=X4_number_of_convenience_stores,
                                   X5_latitude=X5_latitude, X6_longitude=X6_longitude

                                   )

        return JsonResponse({'result': classification, 'No': No,
                             'X1_transaction_date': X1_transaction_date, 'X2_house_age': X2_house_age,
                             'X3_distance_to_nearest_MRT_station': X3_distance_to_nearest_MRT_station,
                             'X4_number_of_convenience_stores': X4_number_of_convenience_stores,
                             'X5_latitude': X5_latitude, 'X6_longitude': X6_longitude},
                            safe=False)


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
