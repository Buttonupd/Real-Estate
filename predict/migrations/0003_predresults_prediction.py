# Generated by Django 3.1 on 2020-08-16 07:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predict', '0002_auto_20200815_1509'),
    ]

    operations = [
        migrations.AddField(
            model_name='predresults',
            name='prediction',
            field=models.IntegerField(default=True, max_length=30),
        ),
    ]