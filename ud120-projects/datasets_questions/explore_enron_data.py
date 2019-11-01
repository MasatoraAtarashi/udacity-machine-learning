#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# count = 0
# for key in enron_data.keys():
#     if enron_data[key]["poi"] == True:
#         count += 1
#
# print(count)
import pprint
# pprint.pprint(enron_data["PRENTICE JAMES"]['total_stock_value'])
# pprint.pprint(enron_data["COLWELL WESLEY"])
# pprint.pprint(enron_data["SKILLING JEFFREY K"]["total_payments"])
# pprint.pprint(enron_data["FASTOW ANDREW S"]["total_payments"])
# pprint.pprint(enron_data["LAY KENNETH L"]["total_payments"])

# count_email = 0
# count_salary = 0
# for key in enron_data.keys():
#     if enron_data[key]["email_address"] == "NaN":
#         count_email += 1
#     if enron_data[key]["salary"] == "NaN":
#         count_salary += 1
#
# print("have email: " + str(len(enron_data) - count_email))
# print("have salary: " + str(len(enron_data) - count_salary))

# count = 0
# for key in enron_data.keys():
#     if enron_data[key]["total_payments"] == "NaN":
#         count += 1
#
# print(count)
# count += 10
# print(count / float(len(enron_data)))

# count = 0
# poi = 0
# for key in enron_data.keys():
#     if enron_data[key]["poi"] == True:
#         poi += 1
#         if enron_data[key]["total_payments"] == "NaN":
#             count += 1
#         else:
#             print(enron_data[key]["total_payments"])

# print(poi)
# print(count)
# print(count / float(poi))

# print(len(enron_data))
