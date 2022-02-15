# Generated by Django 3.1 on 2022-02-14 18:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('prediction', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='studentdetails',
            old_name='nationaity',
            new_name='nationality',
        ),
        migrations.AlterField(
            model_name='studentdetails',
            name='parent_education_status',
            field=models.IntegerField(choices=[(0, 'Below JSC'), (1, 'SSC'), (2, 'HSC'), (3, 'Graduate')]),
        ),
    ]
