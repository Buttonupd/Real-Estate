from django.db import models


class PredResults(models.Model):
    No = models.FloatField(default=True, null=False, blank=False)
    X1_transaction_date = models.FloatField(default=False, null=False)
    X2_house_age = models.FloatField(default=False, null=False)
    X3_distance_to_nearest_MRT_station = models.FloatField(default=False, null=False)
    X4_number_of_convenience_stores = models.FloatField(default=False, null=False)
    X5_latitude = models.FloatField(default=False, null=False)
    X6_longitude = models.FloatField(default=False, null=False)
    Y_house_price_of_unit_area = models.FloatField(default=False, null=False)

    predictions = models.CharField(max_length=30)

    def __str__(self):
        return self.predictions
