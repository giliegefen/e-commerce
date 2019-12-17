import hw3_part1 as st
import Timer


def main():
    """Read the data files with the class ModelData into the a data object"""
    total_time = Timer.Timer('Total running time')
    data_reading = Timer.Timer('Reading Data')
    data = st.ModelData('train_x.csv', 'train_y.csv', 'test_x.csv', 'test_y.csv', 'movie_data.csv')
    data_reading.stop()

    """Calculate the average rating from the train set"""
    r_avg = st.calc_average_rating(data.train_y)

    # *************** Basic Model with average calculated parameters  ***************
    model0_timer = Timer.Timer('Basic Simple Model')
    """Calc the basic parameters vector and run the basic model r_hat = r_avg + b_u + b_i"""
    b0 = st.calc_parameters(r_avg, data.train_x, data.train_y, data)

    """Use the model to inference predictions on the test set"""
    test_predictions = st.model_inference(data.test_x, b0, r_avg, data)
    data.test_x['r_hat'] = test_predictions
    data.test_x['r_hat'] = data.test_x['r_hat'].clip(0.5, 5)

    """Use the model to inference predictions on the train set"""
    train_predictions = st.model_inference(data.train_x, b0, r_avg, data)
    data.train_x['r_hat'] = train_predictions
    data.train_x['r_hat'] = data.train_x['r_hat'].clip(0.5, 5)

    """Calc the RMSE of the train set and the test set, then """
    rmse_timer = Timer.Timer('RMSE calculation')
    print('\nBasic calculated parameters model test RMSE is {:0.4f}'.format(st.calc_error(data.test_x, data.test_y)))
    print('Basic calculated parameters model train RMSE is {:0.4f}\n'.format(st.calc_error(data.train_x, data.train_y)))

    """calc the average RMSE for each movie in each set and plot it as a graph (save the graph)"""
    movie_train_error = st.calc_avg_error(data.train_x, data.train_y)
    st.plot_error_per_movie(movie_train_error)
    movie_test_error = st.calc_avg_error(data.test_x, data.test_y)
    st.plot_error_per_movie(movie_test_error)
    rmse_timer.stop()

    model0_timer.stop()

    # *************** Basic Model with fitted parameters to minimize the RMSE  ***************
    model1_timer = Timer.Timer('Fitted Model without income Construction')

    """Construct vector C from the train set as (r_i - r_avg) """
    c = st.construct_rating_vector(data.train_y, r_avg)

    """Construct the coefficients matrix A"""
    A1 = st.create_coefficient_matrix(data.train_x, data)

    """Fit the parameters vector to minimize the least squares error"""
    b1 = st.fit_parameters(A1, c)

    """Use the basic model r_hat = r_avg + b_u + b_i to inference predictions on the test set"""
    test_predictions = st.model_inference(data.test_x, b1, r_avg, data)
    data.test_x['r_hat'] = test_predictions
    data.test_x['r_hat'] = data.test_x['r_hat'].clip(0.5, 5)

    """Use the basic model r_hat = r_avg + b_u + b_i to inference predictions on the train set"""
    train_predictions = st.model_inference(data.train_x, b1, r_avg, data)
    data.train_x['r_hat'] = train_predictions
    data.train_x['r_hat'] = data.train_x['r_hat'].clip(0.5, 5)

    """Calc the RMSE of the train set and the test set"""
    rmse_timer = Timer.Timer('RMSE calculation')
    print('\nBasic fitted parameters model test RMSE is {:0.4f}'.format(st.calc_error(data.test_x, data.test_y)))
    print('Basic fitted parameters model train RMSE is {:0.4f} \n'.format(st.calc_error(data.train_x, data.train_y)))

    """calc the average RMSE for each movie in each set and plot it as a graph (save the graph)"""
    """Test set"""
    movie_test_error = st.calc_avg_error(data.test_x, data.test_y)
    st.plot_error_per_movie(movie_test_error)
    """Train Set"""
    movie_train_error = st.calc_avg_error(data.train_x, data.train_y)
    st.plot_error_per_movie(movie_train_error)

    rmse_timer.stop()

    model1_timer.stop()

    # *************** Modified Model with income and fitted parameters to minimize the RMSE  ***************
    model2_timer = Timer.Timer('Fitted Model with income')
    """Construct the coefficients matrix A for the model r_hat = r_avg + b_u + movie_income * b_income where 
    b_income stands as the movie income weight parameter"""
    A2 = st.create_coefficient_matrix_with_income(data.train_x, data)

    """Fit the parameters vector to minimize the least squares error"""
    b2 = st.fit_parameters(A2, c)

    """Use the modified model r_hat = r_avg + b_u + b_i to inference predictions on the test set"""
    test_predictions = st.model_inference_with_income(data.test_x, b2, r_avg, data)
    data.test_x['r_hat'] = test_predictions
    # Clipping the predictions to a legal relevant range
    data.test_x['r_hat'] = data.test_x['r_hat'].clip(0.5, 5)

    """Use the modified model r_hat = r_avg + b_u + movie_income * b_income to inference predictions on the train set"""
    train_predictions = st.model_inference_with_income(data.train_x, b2, r_avg, data)
    data.train_x['r_hat'] = train_predictions
    data.train_x['r_hat'] = data.train_x['r_hat'].clip(0.5, 5)

    """Calc the RMSE of the train set and the test set"""
    rmse_timer = Timer.Timer('RMSE calculation')
    print('\nModified fitted parameters model test RMSE is {:0.4f}'.format(st.calc_error(data.test_x, data.test_y)))
    print('Modified fitted parameters model train RMSE is {:0.4f} \n'.format(st.calc_error(data.train_x, data.train_y)))

    """calc the average RMSE for each movie in each set and plot it as a graph (save the graph)"""
    """Test set"""
    movie_test_error = st.calc_avg_error(data.test_x, data.test_y)
    st.plot_error_per_movie(movie_test_error)
    """Train Set"""
    movie_train_error = st.calc_avg_error(data.train_x, data.train_y)
    st.plot_error_per_movie(movie_train_error)

    rmse_timer.stop()

    model2_timer.stop()

    total_time.stop()


if __name__ == '__main__':
    main()
