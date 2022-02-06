from django.db import models


class StudentDetails(models.Model):

    GENDER_CHOICES = [
        (0, 'Male'),
        (1, 'Female'),
    ]

    YES_NO = [
        (0, 'No'),
        (1, 'Yes'),
    ]

    LOW_MEDIUM_HIGH = [
        (0, 'Low'),
        (1, 'Medium'),
        (2, 'High'),
    ]

    PARENTS_EDUCATION_STATUS = [
        (0, 'PSC'),
        (1, 'JSC'),
        (2, 'SSC'),
        (3, 'HSC'),
        (4, 'Above HSC'),
    ]


    created = models.DateTimeField(auto_now_add=True)
    # first_name = models.CharField(max_length=30)
    # last_name = models.CharField(max_length=30)

    #Group - 1
    gender = models.IntegerField(choices=GENDER_CHOICES)
    nationality = models.CharField(max_length=50)
    place_of_birth = models.CharField(max_length=255)
    department = models.CharField(max_length=255)
    year = models.IntegerField()
    institute = models.CharField(max_length=255)

    #Group - 2
    time_of_group_study = models.IntegerField() # In hour
    absent_in_a_semester = models.IntegerField() #In hour
    ask_question_frequently = models.IntegerField(choices=YES_NO)
    use_additional_course_material = models.IntegerField(choices=YES_NO)
    result_of_last_semester = models.DecimalField(max_digits=3, decimal_places=2)
    meet_with_advisor = models.IntegerField(choices=YES_NO)

    #Group - 3
    parent_satisfied = models.IntegerField(choices=YES_NO)
    parent_education_status = models.IntegerField(choices=PARENTS_EDUCATION_STATUS)

    #Group - 4
    amount_of_drop_semester = models.IntegerField()
    drop_reason = models.CharField(max_length=255)
    due_amount = models.DecimalField(max_digits=9, decimal_places=2)

    #Result
    result = models.IntegerField(choices=LOW_MEDIUM_HIGH, blank=True)


    def __str__(self):
        return self.first_name