
from django.db import models


class PredResults(models.Model):

    No = models.IntegerField()
    X1_transaction_date = models.IntegerField()
    X2_house_age = models.IntegerField()
    X3_distance_to_nearest_MRT_station = models.IntegerField
    X4_number_of_convenience_stores = models.IntegerField()
    X5_latitude = models.IntegerField()
    X6_longitude = models.IntegerField()
    Y_house_price_of_unit_area = models.IntegerField()

    def __str__(self):
        return self.classification
