# Generated by Django 3.1 on 2020-08-16 12:46

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0005_auto_20200816_1454'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='predresults',
            name='Y_house_price_of_unit_area',
        ),
    ]
