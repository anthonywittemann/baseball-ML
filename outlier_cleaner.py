#!/usr/bin/python


def outlierCleaner(predictions, predict_x, target_y):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """

    cleaned_data = []
    error = list( (target_y - predictions)**2 )

    cleaned_data = zip(predict_x, target_y, error)
    cleaned_data = sorted(cleaned_data, key = lambda tup: tup[2])
    cleaned_data = cleaned_data[:int(len(cleaned_data) * .8)] # remove 20% - outliers

    return cleaned_data

