from django.shortcuts import render
from .models import StudentDetails
from .serializers import StudentDetailsSerializer
from django.http import Http404, HttpRequest, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from keras.models import model_from_json

import json
import numpy as np



class StudentDropoutPrediction(APIView):
    """
    List all students, or create a new student.
    """
    def get(self, request, format=None):
        student_details = StudentDetails.objects.all()
        serializer = StudentDetailsSerializer(student_details, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        data = request.data #Get all the request data/ form data

        gender = data['gender']
        nationality = data["nationaity"]
        place_of_birth = data["place_of_birth"]
        department = data["department"]
        year = data["year"]
        institute = data["institute"]
        time_of_group_study = data["time_of_group_study"]
        absent_in_a_semester = data["absent_in_a_semester"]
        ask_question_frequently = data["ask_question_frequently"]
        use_additional_course_material = data["use_additional_course_material"]
        result_of_last_semester = data["result_of_last_semester"],
        meet_with_advisor = data["meet_with_advisor"]
        parent_satisfied = data["parent_satisfied"]
        parent_education_status = data["parent_education_status"]
        amount_of_drop_semester = data["amount_of_drop_semester"]
        drop_reason = data["drop_reason"]
        due_amount = data["due_amount"]
        

        #Do Prediction 
        file = open('student_prediction_model.json', 'r')
        model_json = file.read()
        file.close()

        loaded_model = model_from_json(model_json)
        # load_weight = loaded_model.load_weight('student_prediction_model.h5')
        loaded_model.load_weights('student_prediction_model.h5')

        prediction = np.argmax(loaded_model.predict([[
                    # gender,
                    # nationaity,
                    # place_of_birth,
                    # department,
                    # year,
                    # institute,
                    time_of_group_study,
                    absent_in_a_semester,
                    # ask_question_frequently,
                    # use_additional_course_material,
                    # result_of_last_semester,
                    # meet_with_advisor,
                    # parent_satisfied,
                    # parent_education_status,
                    amount_of_drop_semester,
                    # drop_reason,
                    # due_amount
                    ]]),axis=1)

        #Final result will set here
        if(prediction[0]==0):
            data['result'] = prediction[0] # Low
        elif(prediction[0]==1):
            data['result'] = prediction[0] # Medium
        else:
            data['result'] = prediction[0] # High


        serializer = StudentDetailsSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
