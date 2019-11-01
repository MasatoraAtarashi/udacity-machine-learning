#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """


    cleaned_data = []
    ### your code goes here
    # print predictions
    # print ages
    # print net_worths

    # errors = predictions - net_worths
    # errors = map(lambda x: x ** 2, errors)
    # idx = [sorted(range(len(errors)), key=lambda k: errors[k])]
    # errors = sorted(errors)
    # errors = errors[0:(int(len(errors) * 0.9))]
    # ages = ages[idx][0:(int(len(ages) * 0.9))]
    # net_worths = net_worths[idx][0:(int(len(net_worths) * 0.9))]
    # errors = [e for inner_list in errors for e in inner_list]
    # ages = [e for inner_list in ages for e in inner_list]
    # net_worths = [e for inner_list in net_worths for e in inner_list]
    # cleaned_data = [tuple(ages), tuple(net_worths)]
    # print cleaned_data

    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i], net_worths[i], abs(net_worths[i] - predictions[i])))
        cleaned_data = sorted(cleaned_data, key=lambda x:x[2])

    return cleaned_data[:int(len(cleaned_data) * 0.9)]
